
import threading
import pyaudio
from elevenlabs import ElevenLabs

class SpeechOutput:
    """Handles text-to-speech using ElevenLabs."""
    
    def __init__(self, api_key: str, voice_id: str = "nPczCjzI2devNBz1zQrb"):
        """
        Initialize ElevenLabs TTS.
        
        Args:
            api_key: ElevenLabs API key
            voice_id: Voice ID to use (default is "George")
        """
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.audio = pyaudio.PyAudio()
        self._is_speaking = False
        self._stop_signal = False 
        self._is_muted = False    
        self._lock = threading.Lock()

    def stop(self, mute: bool = False):
        """Stops current speech. If mute=True, blocks future speech until unmuted."""
        with self._lock:
            self._stop_signal = True
            if mute:
                self._is_muted = True

    def unmute(self):
        """Allows speech again."""
        with self._lock:
            self._is_muted = False

    def speak(self, text: str, wait: bool = True, ignore_mute: bool = False):
        with self._lock:
            if self._is_muted and not ignore_mute:
                print(f"🤫 Skipped speech (Muted): '{text}'")
                return None
            
            self._stop_signal = False

        def _speak_thread():
            stream = None
            try:
                audio_stream = self.client.text_to_speech.stream(
                    text=text, 
                    voice_id=self.voice_id, 
                    model_id="eleven_multilingual_v2", 
                    output_format="pcm_22050"
                )

                stream = self.audio.open(
                    format=pyaudio.paInt16, channels=1, rate=22050, 
                    output=True, frames_per_buffer=1024
                )

                for chunk in audio_stream:
                    with self._lock:
                        if self._stop_signal:
                            break
                    if chunk:
                        stream.write(chunk)
            except Exception as e:
                print(f"Speech error: {e}")
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()

        if wait:
            _speak_thread()
            return None
        else:
            thread = threading.Thread(target=_speak_thread, daemon=True)
            thread.start()
            return thread

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._lock:
            return self._is_speaking

    def cleanup(self):
        """Clean up audio resources."""
        self.stop()
        if self.audio:
            self.audio.terminate()