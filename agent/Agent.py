import queue
import threading
from typing import Optional
from dataclasses import dataclass


from openai import OpenAI
import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
from pydantic import BaseModel
from agent.SpeechOutput import SpeechOutput
from agent.state import State


class OutputFormat(BaseModel):
    explanation: str
    code: str


class IntentFormat(BaseModel):
    understanding: str  # What the user actually wants
    approach: str       # High-level strategy (1-2 sentences)

CONCISE_INSTRUCTION = """\

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
send_msg_to_user("$279.49")
"

IMPORTANT RULES:
- Only send_msg_to_user when you have the COMPLETE answer to the goal.
- If information is missing, navigate/click/search to find it first.
- Never ask the user "if you want" or "would you like" - just do it.
- Never send partial answers or say "I couldn't find" - keep trying.


Make sure you have an explanation and an action.
"""


@dataclass
class AgentCommand:
    goal: str
    on_complete: Optional[callable] = None


class BrowserAgent:
    def __init__(self, llm: str, client: OpenAI, speech: Optional[SpeechOutput] = None):
        self.client = client
        self.llm = llm
        self.speech = speech
        self.env = None
        self.action_space = HighLevelActionSet(
            subsets=['chat', 'nav', 'bid'], 
            strict=False, 
            multiaction=True
        )
        self.obs = None
        
        # Command queue for thread-safe communication
        self._command_queue = queue.Queue()
        self._stop_requested = False
        self._is_running = False
        self._lock = threading.Lock()
        
        # Start dedicated agent thread
        self._agent_thread = threading.Thread(target=self._agent_loop, daemon=True)
        self._agent_thread.start()

    def _agent_loop(self):
        """Main loop running on dedicated thread - browser lives here."""
        print("🌐 Starting browser...")
        self.env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank'},
            headless=False,
            tags_to_mark='all'
        )
        self.obs, _ = self.env.reset()
        print("✅ Browser ready!")

        while True:
            try:
                try:
                    command = self._command_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if command is None:  # Shutdown signal
                    break

                self._execute_goal(command.goal, command.on_complete)

            except Exception as e:
                print(f"❌ Agent loop error: {e}")

        if self.env:
            self.env.close()
    
    def get_intent(self, goal):
        response = self.client.responses.parse(
            model=self.llm,
            input=[{
                "role": "system", 
                "content": "You're about to help with a web task. Briefly describe your understanding and approach. Be conversational, like you're thinking out loud."
            }, {
                "role": "user", 
                "content": f"Goal: {goal}"
            }],
            text_format=IntentFormat,
        )
        return response.output_parsed

    def _execute_goal(self, goal: str, on_complete: Optional[callable] = None, max_steps: int = 50):
        """Execute a goal - runs on agent thread."""
        with self._lock:
            self._stop_requested = False
            self._is_running = True

        state = State(goal=goal)
        intent = self.get_intent(goal)
        state.set_intent(intent)
        state.set_obs(self.obs)
        system_message = self.get_system_message(goal)

        # Announce the goal
        print(f"\n🎯 New Goal: {goal}\n")
        if self.speech:
            self.speech.speak(intent.approach, wait=True)
            

        try:
            for step_num in range(max_steps):
                with self._lock:
                    if self._stop_requested:
                        print("\n🛑 Task interrupted by new voice command.")
                        if self.speech:
                            self.speech.stop()
                        return

                print(f"\n--- Step {step_num + 1}/{max_steps} ---")

                if state.get_errors() > 5:
                    if self.speech:
                        self.speech.speak("Too many errors. Stopping task.", wait=True)
                    break

                try:
                    done = self.step(state, goal, system_message)
                    if done:
                        print("\n✅ Goal accomplished!")
                        if on_complete:
                            on_complete(True)
                        return
                except Exception as e:
                    print(f"⚠️  Step error: {e}")
                    state.record_error()
                    try:
                        self.obs, _ = self.env.reset()
                        state.set_obs(self.obs)
                    except Exception as reset_error:
                        print(f"❌ Could not recover: {reset_error}")
                        if self.speech:
                            self.speech.speak("Could not recover from error.", wait=True)
                        break

            print(f"\n⏰ Reached max steps ({max_steps})")
            if self.speech:
                self.speech.speak("Reached maximum steps without completing the goal.", wait=True)
            
        finally:
            with self._lock:
                self._is_running = False
            if on_complete:
                on_complete(False)
            print("\n🎤 Ready for next voice command. Press Option to speak...")

    def step(self, state: State, goal: str, system_message: str) -> bool:
        """Execute a single step with speech. Returns True if task is complete."""
        prompt = self.get_prompt(state, goal)

        response = self.client.responses.parse(
            model=self.llm,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            text_format=OutputFormat,
        )

        event = response.output_parsed
        explanation = event.explanation
        action = event.code.strip()

        print(f"💭 Thought: {explanation}")
        print(f"🎬 Action: {action}")

        # Check for incomplete send_msg_to_user
        if "send_msg_to_user" in action:
            incomplete_phrases = [
                "if you want", "would you like", "do you want",
                "let me know", "I can't see", "I couldn't find", "should I"
            ]
            if any(phrase in action.lower() for phrase in incomplete_phrases):
                print("⚠️  Rejecting incomplete response - need more info")
                if self.speech:
                    self.speech.speak("I need more information. Trying again.", wait=True)
                state.record_error()
                return False

        # Start speaking the explanation asynchronously
        speech_thread = None
        if self.speech:
            speech_thread = self.speech.speak_async(explanation)

        # Execute the action while speech plays
        state.add_action(action)

        try:
            self.obs, reward, done, truncated, info = self.env.step(action)
            state.set_obs(self.obs)

            error = self.obs.get('last_action_error', '')
            if error:
                print(f"⚠️  Action error: {error}")
                state.record_error()
            else:
                state.record_success()

            # Wait for speech to complete before next step
            if speech_thread and speech_thread.is_alive():
                speech_thread.join()

            # Check if done
            if done or "send_msg_to_user" in action:
                # Extract and speak the message sent to user
                if "send_msg_to_user" in action and self.speech:
                    try:
                        # Extract message from action like: send_msg_to_user("message here")
                        import re
                        match = re.search(r'send_msg_to_user\(["\'](.+?)["\']\)', action, re.DOTALL)
                        if match:
                            message = match.group(1)
                            self.speech.speak(f"Here's what I found: {message}", wait=True)
                    except:
                        pass
                return True

            return False

        except Exception as e:
            # Wait for speech even on error
            if speech_thread and speech_thread.is_alive():
                speech_thread.join()
            print(f"❌ Environment error: {e}")
            state.record_error()
            raise

    def run(self, goal: str, max_steps: int = 50, on_complete: Optional[callable] = None):
        """Queue a goal to be executed (thread-safe)."""
        # Stop any current speech
        if self.speech:
            self.speech.stop()

        # Clear any pending commands
        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

        self._command_queue.put(AgentCommand(goal=goal, on_complete=on_complete))

    def stop(self):
        """Request current task to stop."""
        with self._lock:
            self._stop_requested = True
        if self.speech:
            self.speech.stop()
        print("\n⏹️  Stop requested...")

    def is_running(self) -> bool:
        with self._lock:
            return self._is_running

    def get_system_message(self, goal: str) -> str:
        return f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

IMPORTANT:
- Complete the ENTIRE task before sending a message to the user.
- Do NOT ask the user for permission or clarification - just complete the task.
- Only use send_msg_to_user() when you have ALL the information requested.
- If you need more information, take actions to get it (click, navigate, search).

When explaining your actions, speak naturally like you're thinking out loud:
- BAD: "In order to accomplish my goal I need to click the search button"
- GOOD: "Let me search for that..." or "I'll click on this coffee shop to see the reviews"

Keep explanations to ONE short sentence. Sound like a helpful friend, not a robot.

# Goal:
{goal}

# Action Space
{self.action_space.describe(with_long_description=False, with_examples=True)}
"""
    def get_prompt(self, state: State, goal: str) -> str:
        if state.obs is None:
            return ""

        obs = state.get_obs()
        cur_axtree_txt = flatten_axtree_to_str(obs['axtree_object'])
        cur_url = obs['url']
        error_prefix = obs.get('last_action_error', '')

        prompt = f"""
{f'# Previous Action Error: {error_prefix}' if error_prefix else ''}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

# Previous Actions:
{' | '.join(state.get_actions()[-5:]) if state.get_actions() else 'None'}



""".strip()
        prompt += f"""Intent: \n 
{state.get_intent().approach}"""

        prompt += CONCISE_INSTRUCTION

        if state.get_errors() > 3:
            prompt += """

⚠️  WARNING: You are failing this task. Try a completely different approach.
DO NOT send_msg_to_user until you have ALL requested information.
"""
        return prompt

    def cleanup(self):
        """Shutdown the agent."""
        self._command_queue.put(None)
        self._agent_thread.join(timeout=5.0)
