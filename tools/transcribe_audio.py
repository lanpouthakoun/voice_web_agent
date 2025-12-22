#!/usr/bin/env python3
"""
Voice transcription tool that records audio when Option key is pressed
and transcribes it using OpenAI's GPT-4o mini Transcribe API.
"""

import os
import sys
import io
import queue
import threading
import asyncio

try:
    import speech_recognition as sr
    import pyaudio
    from openai import OpenAI
    from pynput import keyboard
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


class VoiceTranscriber:
    def __init__(self, api_key, mic_index=None):
        self.client = OpenAI(api_key=api_key)
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.audio_queue = queue.Queue()
        self.mic_index = mic_index
        self.is_recording = False
        self.recorded_frames = []
        self.option_pressed = False
        self.audio = None
        self.stream = None
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
    def _record_audio_directly(self):
        """Record audio directly while Option is pressed using pyaudio."""
        if not self.audio:
            self.audio = pyaudio.PyAudio()
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.mic_index
            )
            
            self.recorded_frames = []
            
            # Record while Option is pressed
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.recorded_frames.append(data)
                except Exception as e:
                    print(f"[Record Error]: {e}")
                    break
            
            # Stop and close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            # Return combined audio data
            if self.recorded_frames:
                return b''.join(self.recorded_frames)
            return None
            
        except Exception as e:
            print(f"[Record Error]: {e}")
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            return None

    def is_valid_logprobs(self, logprobs, threshold=-0.1) -> bool:
        """Check if transcription logprobs are valid."""
        if not logprobs:
            return False
        avg_logprob = sum(lp.logprob for lp in logprobs) / len(logprobs)
        return avg_logprob > threshold

    async def _process_audio(self):
        """Process recorded audio and transcribe."""
        while True:
            try:
                # Wait for audio to be ready
                if not self.audio_queue.empty():
                    wav_data = self.audio_queue.get()
                    
                    wav_stream = io.BytesIO(wav_data)
                    wav_stream.name = "audio.wav"
                    wav_stream.seek(0)

                    try:
                        response = self.client.audio.transcriptions.create(
                            model="gpt-4o-mini-transcribe",
                            file=wav_stream,
                            temperature=0.2,
                            response_format="json",
                            include=["logprobs"],
                            timeout=10,
                        )
                        
                        is_valid = self.is_valid_logprobs(response.logprobs)
                        print(f"\n✅ Is valid transcript: {is_valid}")
                        print("="*60)
                        print("📝 TRANSCRIPTION:")
                        print("="*60)
                        print(response.text)
                        print("="*60 + "\n")
                        
                    except Exception as e:
                        print(f"\n❌ [Transcription Error]: {e}\n")
                        
            except queue.Empty:
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                print(f"[Process Error]: {e}")
                await asyncio.sleep(0.1)

    def on_option_press(self):
        """Handle Option key press - start recording."""
        if not self.option_pressed:
            self.option_pressed = True
            self.is_recording = True
            print("\n🎤 Recording... (release Option to stop)")
            
            # Start recording in a separate thread
            def record():
                raw_audio = self._record_audio_directly()
                if raw_audio:
                    # Convert to WAV format using speech_recognition
                    try:
                        audio_data = sr.AudioData(
                            raw_audio,
                            self.sample_rate,
                            2  # 16-bit = 2 bytes per sample
                        )
                        wav_data = audio_data.get_wav_data()
                        self.audio_queue.put(wav_data)
                    except Exception as e:
                        print(f"❌ Error processing audio: {e}")
            
            self.recording_thread = threading.Thread(target=record, daemon=True)
            self.recording_thread.start()

    def on_option_release(self):
        """Handle Option key release - stop recording and transcribe."""
        if self.option_pressed:
            self.option_pressed = False
            self.is_recording = False
            
            # Wait for recording thread to finish
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
            
            print("⏹️  Recording stopped. Transcribing...")

    async def _record_loop(self):
        """Initialize and keep the loop alive."""
        print("✅ Ready. Press and hold Option key to record...\n")
        
        # Keep the loop alive
        while True:
            await asyncio.sleep(1)
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None

    def start_keyboard_listener(self, loop):
        """Start keyboard listener in a separate thread."""
        def on_press(key):
            try:
                if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    self.on_option_press()
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    self.on_option_release()
                if key == keyboard.Key.esc:
                    return False
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            try:
                listener.join()
            except KeyboardInterrupt:
                pass

    async def run(self):
        """Run recording and transcription tasks concurrently."""
        # Start keyboard listener in a separate thread
        keyboard_thread = threading.Thread(
            target=self.start_keyboard_listener,
            args=(asyncio.get_event_loop(),),
            daemon=True
        )
        keyboard_thread.start()
        
        # Run async tasks
        await asyncio.gather(self._record_loop(), self._process_audio())


def list_microphones():
    """List available microphones."""
    try:
        available_mics = sr.Microphone.list_working_microphones()
        available_mics = list(set(available_mics))
        print("Available microphones:")
        for i, mic in enumerate(available_mics):
            print(f"  [{i}] {mic}")
        return available_mics
    except Exception as e:
        print(f"Error listing microphones: {e}")
        return []


def main():
    # Get the script's directory to find .env file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(script_dir, '.env')
    
    # Check if .env file exists
    env_exists = os.path.exists(env_path)
    
    # Load environment variables from .env file, overriding existing env vars
    # This ensures .env file takes precedence over system environment variables
    if env_exists:
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded .env file from: {env_path}")
    else:
        # Try to load from current directory as fallback
        load_dotenv(override=True)
        if os.path.exists('.env'):
            print("✅ Loaded .env file from current directory")
        else:
            print("⚠️  No .env file found. Using environment variables or will prompt.")
    
    # Get API key from environment variable or user input
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in .env file or environment variables.")
        api_key = input("Please enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ API key is required. Exiting.")
            print(f"💡 Tip: Create a .env file at {env_path} with: OPENAI_API_KEY=your-api-key-here")
            sys.exit(1)
    else:
        # Show first and last few characters for verification (don't print full key)
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✅ Using API key: {masked_key}")
    
    # Optional: List microphones to help user find their mic index
    mic_index = None
    if len(sys.argv) > 1:
        try:
            mic_index = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid microphone index: {sys.argv[1]}. Using default.")
    
    # Uncomment to list available microphones
    # list_microphones()
    
    print("🎙️  Voice Transcriber started!")
    print("Press and hold Option (Alt) key to record audio")
    print("Release Option key to stop recording and transcribe")
    print("Press Escape or Ctrl+C to exit\n")
    
    transcriber = VoiceTranscriber(api_key, mic_index=mic_index)
    
    try:
        asyncio.run(transcriber.run())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        transcriber.cleanup()


if __name__ == "__main__":
    main()
