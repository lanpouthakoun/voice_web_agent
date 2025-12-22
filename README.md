# Voice Web Agent

A real-time voice transcription tool that streams audio when you press the Option key and transcribes it using OpenAI's Realtime API with GPT-4o mini Transcribe model.

## Features

- 🎙️ Press and hold Option key to start streaming audio
- ⏹️ Release Option key to stop streaming and get transcription
- 📝 Real-time streaming transcription using OpenAI's Realtime API
- 🔄 Live incremental transcription updates as you speak
- 🔊 Automatic microphone access handling

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for macOS users:** You may need to install PortAudio first for `pyaudio`:
```bash
brew install portaudio
```

### 2. Set Your API Key

You can provide your OpenAI API key in one of three ways:

**Option 1: .env file (Recommended)**
Create a `.env` file in the project directory:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**Option 2: Environment Variable**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option 3: Enter when prompted**
The script will ask for your API key if it's not found in `.env` or environment variables.

### 3. Grant Permissions

On macOS, you'll need to grant:
- **Microphone access** - The app will request this when you first run it
- **Accessibility permissions** - For keyboard monitoring (System Settings > Privacy & Security > Accessibility)

## Usage

Run the script:
```bash
python transcribe_audio.py
```

Then:
1. Press and **hold** the **Option** (Alt) key to start recording
2. Speak into your microphone
3. **Release** the Option key to stop recording and get the transcription
4. Press **Escape** or **Ctrl+C** to exit

## How It Works

- The script uses `pyaudio` to capture audio from your microphone
- It monitors for Option key presses using `pynput`
- When Option is pressed, it starts recording audio chunks
- When Option is released, it stops recording and sends the audio to OpenAI's transcription API
- Uses GPT-4o mini Transcribe model for efficient and accurate transcription
- The transcribed text is displayed with logprobs validation

## Troubleshooting

- **"Missing required package" error**: Make sure you've installed all dependencies with `pip install -r requirements.txt`
- **No audio recording**: Check that microphone permissions are granted in System Settings
- **Keyboard not detected**: Make sure Accessibility permissions are granted for the terminal/Python
- **PortAudio errors on macOS**: Install PortAudio with `brew install portaudio`