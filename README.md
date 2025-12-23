# Voice - Voice Browser Agent

## Setup

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- macOS/Linux system (for pyaudio compatibility)

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd voice_web_agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 2: Install System Dependencies (macOS)

For PyAudio to work on macOS, you may need to install PortAudio:

```bash
# Using Homebrew
brew install portaudio
```

### Step 3: Install Python Dependencies

```bash
# Make sure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root directory:

```bash
# Create .env file
touch .env
```

Add the following to `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=nPczCjzI2devNBz1zQrb  # Optional, defaults to this value
```

Replace `your_openai_api_key_here` and `your_elevenlabs_api_key_here` with your actual API keys.

### Step 5: Verify Installation

```bash
# Run the application
python run.py
```

The application will prompt you for API keys if they're not found in the `.env` file.
The applciation will open up a Playwright Chromium window and you will be told how to add voice commands.

## Agent Architecture Analysis


The agent consists of three main components: speech-to-text, a browser agent, and text-to-speech.

---

### 1. Speech-to-Text

- **Model**: GPT-4o-mini-transcribe (OpenAI API)
- **Rationale**: Best accuracy among tested options; familiarity with OpenAI API
- **Input Mode**: Push-to-talk
  - Prevents race conditions, allows user speech to take precedence over agent speech
  - Allows user speech and agent speech to run in parallel with browser actions

#### Other Options Considered

| Architecture | Why I Chose/Rejected |
|--------------|---------------------|
| **Whisper** |  Not as accurate |
| **GPT-4o-mini-transcribe**  | More Accurate and Allowed for logprob confidence scores |

---


### 2. Browser Agent

#### Browser Layer
- **Tool**: PlaywrightMCP
- **Rationale**: Returns DOM directly, avoiding latency from screenshot-based approaches

##### Other Options Considered

| Tool | Approach | Pros | Cons | Why I Chose/Rejected |
|------|----------|------|------|---------------------|
| **PlaywrightMCP** | DOM/AXTree access | Low latency, structured element IDs, no vision model needed | No visual reasoning, can't handle canvas/images | ✅ Chosen — speed was priority |
| **Claude Computer Use** | Screenshot + vision | Can reason about layout, handles any UI | 200-500ms per frame, requires vision model calls | Too slow for real-time voice |
| **Qwen Computer Use** | Screenshot + vision | Open weights, can self-host | Same latency issues as Claude |  Same tradeoff |
| **BrowserUse** | High-level browser API | Simple API, good abstractions | No internal control over agent | I wanted to create a custom agent |
| **Browser Base** | Cloud browser infrastructure | Scalable, no local browser needed | Added network latency, cost | Local browser faster for demos |
---

#### Orchestration
- **Framework**: BrowserGym with PlaywrightMCP
- **Rationale**: Enables future evaluation testing via Browser Company Evaluation System

#### Agent Architecture
- **Design**: Custom agent based on Event Stream architecture (OpenHands paper) with persistent written memory (like notes) and light planning (not gated by plan)
- **Rationale**:
  - Browser environments are highly variable; planning architectures are less effective (so we don't gate with the plan)
  - OpenHands-based agents perform top-3 on SWE-Bench Verified
  - Adapted this architecture for browser-specific tasks


##### Additional Architecture Notes

- Implemented Loop Detection (when the agent was stuck it was detected and solved for)
- Dual-mode interruption: Option key starts new goal, Command key modifies current goal mid-task
- Syntax Detection (LLM might return poor output, this was accounted for)

##### Other Options Considered

| Architecture | Description | Why I Chose/Rejected |
|--------------|-------------|---------------------|
| **Planning Agent** | Generate full plan, then execute |  Plans go stale after one action in dynamic DOMs |
| **Event Stream (OpenHands)** | One action → observe → repeat | Tight feedback loops handle page changes |
| **Tree Search (MCTS)** | Explore action branches | Too slow for real-time voice interaction |

---

### 3. Text-to-Speech

- **API**: ElevenLabs (default voice)
- **Execution**: Runs in parallel with browser agent actions
- **Behavior**:
  - Each action is paired with an explanation
  - Explanations are skipped if speech is still playing, but the paired actions are still executed
  - Actions are individual actions i.e. clicking a single dropdown and clicking an item within that dropdown are separate api calls (to maintain accuracy), so explanations naturally compound

#### Other Options Considered

| Architecture | Why I Chose/Rejected |
|--------------|---------------------|
| **ElevenLabsr** |  Natural voice quality, streaming support, reasonable cost |
| **OpenAI TTS**  | Lower voice quality, no streaming in real-time |
---

## Limitations

 - I based the agent on OpenHands architecture, would be interesting to see if the performance would increase if we included planning features as well.
 - To decrease latency, some explanations are skipped, sometimes this leads to page changes that are left unexplained
 - Each browser action is a single action (no multiaction), this is to maintain the best possible accuracy within tasks; however, it leads to slower output
 - By separating browser actions, it leads to more api calls, to make this more cost effective, I would need to rely less on single actions or use a cheaper model (mostly tested with gpt-5.2)
 - Some of the explanations aren't directly suited for voice agents. Could be better prompted.


## Future Work
  - Hybrid vision fallback for canvas/complex widgets
  - Voice activity detection to replace push-to-talk
  - Multi-action batching for simple sequences
  - 
## Demo
[Watch the demo](https://www.loom.com/share/c97a9b31ff374b06bc4ec9792dfcc870)
