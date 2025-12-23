import io
import queue
import threading
import asyncio
import speech_recognition as sr
import pyaudio
from openai import OpenAI
from pynput import keyboard
from typing import Optional, Tuple
from agent.SpeechOutput import SpeechOutput
from agent.Agent import BrowserAgent

class VoiceControlledAgent:
    def __init__(
        self, 
        openai_api_key: str, 
        elevenlabs_api_key: str,
        llm: str = "gpt-4o", 
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        mic_index: Optional[int] = None
    ):
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.speech = SpeechOutput(api_key=elevenlabs_api_key, voice_id=voice_id)
        
        self.agent = BrowserAgent(llm=llm, client=self.openai_client, speech=self.speech)
        
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.mic_index = mic_index
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
        self.is_recording = False
        self.option_pressed = False
        self.command_pressed = False
        self.recorded_frames = []
        self.audio = None
        self.stream = None
        
        # Queue now holds a tuple: (wav_data, mode)
        # mode will be either "new" or "add"
        self.transcription_queue = queue.Queue()

    def record_audio_directly(self) -> Optional[bytes]:
        """Record audio directly while Option is pressed."""
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

            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.recorded_frames.append(data)
                except Exception as e:
                    print(f"[Record Error]: {e}")
                    break

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

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

    def transcribe_audio(self, wav_data: bytes) -> Optional[str]:
        """Transcribe audio data to text."""
        wav_stream = io.BytesIO(wav_data)
        wav_stream.name = "audio.wav"
        wav_stream.seek(0)

        try:
            response = self.openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=wav_stream,
                temperature=0.2,
                response_format="json",
                include=["logprobs"],
                timeout=10,
            )
            if response.logprobs:
                avg_logprob = sum(lp.logprob for lp in response.logprobs) / len(response.logprobs)
                if avg_logprob < -0.5:
                    print("⚠️  Low confidence transcription, please try again.")
                    self.speech.speak("I didn't catch that. Please try again.", wait=True)
                    return None

            return response.text.strip()

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None

    def on_option_press(self):
        """Handle Option key press - Start NEW goal."""
        if not self.option_pressed:
            self.option_pressed = True
            self.is_recording = True
            
            if self.agent.is_running():
                print("\n Interrupting current task for NEW goal...")
                self.agent.stop()

            self.speech.stop(mute = True)
            print("\n 🆕 Recording NEW GOAL... (release Option to stop)")

            def record_and_process():
                raw_audio = self.record_audio_directly()
                if raw_audio:
                    try:
                        audio_data = sr.AudioData(raw_audio, self.sample_rate, 2)
                        wav_data = audio_data.get_wav_data()
                        self.transcription_queue.put((wav_data, "new"))
                    except Exception as e:
                        print(f"❌ Error processing audio: {e}")

            self.recording_thread = threading.Thread(target=record_and_process, daemon=True)
            self.recording_thread.start()

    def on_option_release(self):
        """Handle Option key release."""
        if self.option_pressed:
            self.option_pressed = False
            self.is_recording = False

            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)

            print("⏹️  Recording stopped. Processing New Goal...")
            self.speech.unmute()

    def on_command_press(self):
        """Handle Command key press - ADD to existing goal."""
        if not self.command_pressed:
            self.command_pressed = True
            self.is_recording = True

            self.speech.stop(mute = True)

            print("\n Recording ADDITION to goal... (release Command to stop)")

            def record_and_process():
                raw_audio = self.record_audio_directly()
                if raw_audio:
                    try:
                        audio_data = sr.AudioData(raw_audio, self.sample_rate, 2)
                        wav_data = audio_data.get_wav_data()
                        self.transcription_queue.put((wav_data, "add"))
                    except Exception as e:
                        print(f"❌ Error processing audio: {e}")

            self.recording_thread = threading.Thread(target=record_and_process, daemon=True)
            self.recording_thread.start()

    def on_command_release(self):
        """Handle Command key release."""
        if self.command_pressed:
            self.command_pressed = False
            self.is_recording = False
            
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)

            print("⏹️  Recording stopped. Processing Addition...")
            

    async def process_transcriptions(self):
        """Process transcription queue and dispatch to agent based on mode."""
        while True:
            try:
                if not self.transcription_queue.empty():
                    wav_data, mode = self.transcription_queue.get()
                    
                    text = self.transcribe_audio(wav_data)

                    if text:
                        print(f"\n📝 Transcription ({mode}): {text}")
                        print("=" * 60)
                        
                        if mode == "new":
                            if self.agent.is_running():
                                self.agent.stop()
                            self.agent.run(text, max_steps = 50)
                        
                        elif mode == "add":
                            print(f"Injecting into running agent: {text}")
                            self.agent.add_goal(text, max_steps = 50)
                            
                    else:
                        print("\n⚠️  No valid transcription. Try again.")
                        print("🎤 Ready.")

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"[Process Error]: {e}")
                await asyncio.sleep(0.1)

    async def keep_alive(self):
        """Keep the main loop alive."""
        print("\n" + "=" * 60)
        print("🎙️  Voice-Controlled Browser Agent Ready!")
        print("=" * 60)
        print("• Hold OPTION (Alt) -> NEW Goal")
        print("• Hold COMMAND (Cmd/Win) -> ADD to Goal")
        print("• Press Escape or Ctrl+C to exit")
        print("=" * 60 + "\n")
        
        self.speech.speak("Agent ready.", wait=True)
        
        while True:
            await asyncio.sleep(1)

    def start_keyboard_listener(self):
        """Start keyboard listener for Option and Command keys."""
        def on_press(key):
            try:
                if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    self.on_option_press()
                
                if key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                    self.on_command_press()
            except AttributeError:
                pass

        def on_release(key):
            try:
                if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                    self.on_option_release()
                
                if key == keyboard.Key.cmd or key == keyboard.Key.cmd_l or key == keyboard.Key.cmd_r:
                    self.on_command_release()
                
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
        """Run the voice-controlled agent."""
        keyboard_thread = threading.Thread(
            target=self.start_keyboard_listener,
            daemon=True
        )
        keyboard_thread.start()

        await asyncio.gather(
            self.keep_alive(),
            self.process_transcriptions()
        )

    def cleanup(self):
        """Clean up all resources."""
        if self.agent.is_running():
            self.agent.stop()

        self.agent.cleanup()
        self.speech.cleanup()

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.audio:
            self.audio.terminate()
            self.audio = None