import queue
import threading
from typing import Optional, Tuple
from dataclasses import dataclass


from openai import OpenAI
import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
from pydantic import BaseModel
from agent.SpeechOutput import SpeechOutput
from agent.state import *
from agent.Action import *


class ReflectionFormat(BaseModel):
    what_went_wrong: str
    new_approach: str


class OutputFormat(BaseModel):
    explanation: str
    code: str


class IntentFormat(BaseModel):
    understanding: str
    approach: str


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
        
        self._command_queue = queue.Queue()
        self._stop_requested = False
        self._is_running = False
        self._lock = threading.Lock()
        
        self._agent_thread = threading.Thread(target=self._agent_loop, daemon=True)
        self._agent_thread.start()

    def _agent_loop(self):
        """Main loop running on dedicated thread - browser lives here."""
        print("🌐 Starting browser...")
        self.env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank'},
            headless=False,
            tags_to_mark='standard_html',
            timeout=3000
        )
        self.obs, _ = self.env.reset()
        print("✅ Browser ready!")

        while True:
            try:
                try:
                    command = self._command_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if command is None:
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
                "content": """You're about to help with a web task. Briefly describe your understanding and approach.

Be conversational. Do NOT ask the user for permission - just complete the task."""
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
        state.intent = intent
        
        system_message = self.get_system_message(goal)

        if self.speech:
            self.speech.speak(intent.approach, wait=True)

        if self.obs:
            initial_obs_event = state.update_from_observation(self.obs)
            state.add_event(initial_obs_event)
            
        try:
            for step_num in range(max_steps):
                with self._lock:
                    if self._stop_requested:
                        print("\n🛑 Task interrupted by new voice command.")
                        if self.speech:
                            self.speech.stop()
                        return

                print(f"\n--- Step {step_num + 1}/{max_steps} ---")

                if state.consecutive_errors > 5:
                    if self.speech:
                        self.speech.speak("I'm having too much trouble. I'll stop here.", wait=True)
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
                    error_event = BrowserObservation(
                        source=EventSource.ENVIRONMENT,
                        error=f"System Exception: {str(e)}",
                        last_action_success=False
                    )
                    state.add_event(error_event)
                    
                    try:
                        self.obs, _ = self.env.reset()
                        reset_event = state.update_from_observation(self.obs)
                        state.add_event(reset_event)
                    except Exception as reset_error:
                        print(f"❌ Could not recover: {reset_error}")
                        if self.speech:
                            self.speech.speak("I cannot recover the browser.", wait=True)
                        break

            print(f"\n⏰ Reached max steps ({max_steps})")
            if self.speech:
                self.speech.speak("I reached the maximum number of steps without finishing.", wait=True)
            
        finally:
            with self._lock:
                self._is_running = False
            if on_complete:
                on_complete(False)
            print("\n🎤 Ready for next voice command. Press Option to speak...")

    def step(self, state: State, goal: str, system_message: str) -> bool:
        """Execute a single step with speech. Returns True if task is complete."""
        
        prompt = self.get_prompt(state, goal)
        
        # Debug: print the history context
        print("=" * 50)
        print("HISTORY CONTEXT SENT TO LLM:")
        print(state.view.get_prompt_context(max_events=10))
        print("=" * 50)

        response = self.client.responses.parse(
            model=self.llm,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            text_format=OutputFormat,
        )

        event_data = response.output_parsed
        explanation = event_data.explanation
        action_code = event_data.code.strip()

        print(f"💭 Thought: {explanation}")
        print(f"🎬 Action: {action_code}")

        # Validate and fix common action syntax errors
        action_code, validation_error = self._validate_action(action_code)
        if validation_error:
            print(f"⚠️ Fixed action syntax: {validation_error}")

        action_event = BrowserAction(
            source=EventSource.AGENT,
            thought=explanation,
            code=action_code
        )
        state.add_event(action_event)

        if "send_msg_to_user" in action_code:
            incomplete_phrases = [
                "if you want", "would you like", "do you want",
                "let me know", "I can't see", "I couldn't find", "should I"
            ]
            if any(phrase in action_code.lower() for phrase in incomplete_phrases):
                print("⚠️  Rejecting incomplete response")
                if self.speech:
                    self.speech.speak("I need to find more information first.", wait=True)
                
                error_event = BrowserObservation(
                    source=EventSource.ENVIRONMENT,
                    error="Action Rejected: You tried to ask the user a question. Complete the task yourself.",
                    last_action_success=False
                )
                state.add_event(error_event)
                return False

        speech_thread = None
        if self.speech:
            speech_thread = self.speech.speak(explanation, wait = True)

        try:
            self.obs, reward, done, truncated, info = self.env.step(action_code)

            obs_event = state.update_from_observation(self.obs)
            state.add_event(obs_event)

            if obs_event.error:
                print(f"⚠️ Action error: {obs_event.error}")
            elif not obs_event.last_action_success:
                 print("⚠️ No page change detected")
            else:
                 print("✅ Page updated")

            # if speech_thread and speech_thread.is_alive():
            #     speech_thread.join()

            if done or "send_msg_to_user" in action_code:
                if "send_msg_to_user" in action_code and self.speech:
                    try:
                        import re
                        match = re.search(r'send_msg_to_user\(["\'](.+?)["\']\)', action_code, re.DOTALL)
                        if match:
                            message = match.group(1)
                            self.speech.speak(f"Here is what I found: {message}", wait=True)
                    except:
                        pass
                return True

            return False

        except Exception as e:
            if speech_thread and speech_thread.is_alive():
                speech_thread.join()
            print(f"❌ Environment error: {e}")
            raise e

    def run(self, goal: str, max_steps: int = 50, on_complete: Optional[callable] = None):
        """Queue a goal to be executed (thread-safe)."""
        if self.speech:
            self.speech.stop()

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
# Browser Automation Agent

You control a web browser ONE ACTION AT A TIME. After each action, you see the result.

## Actions - EXACT SYNTAX (ONE per turn)
```
fill('bid', 'text')        # Types text into field. DOES NOT SUBMIT.
click('bid')               # Clicks an element
press('Enter')             # Press a key (after fill to submit)
goto('https://url.com')    # Navigate to URL
scroll(0, 500)             # Scroll down (positive y = down)
send_msg_to_user('msg')    # Return final answer to user
```

## SYNTAX RULES - IMPORTANT
- bid and text must be in QUOTES: fill('123', 'hello') ✓
- fill() takes EXACTLY 2 arguments: fill('bid', 'text') ✓
- DO NOT add extra parameters: fill('123', 'hello', true) ✗ WRONG
- scroll() takes 2 numbers (no quotes): scroll(0, 500) ✓

## Workflow Patterns
**Search:** fill('searchbox_id', 'query') → click('search_btn_id') or press('Enter')
**Form:** fill('field1', 'value1') → fill('field2', 'value2') → click('submit_btn')

## CRITICAL RULES
1. fill() only types text - you MUST click a button or press Enter to submit
2. Use exact bid values from the accessibility tree
3. If action fails, try a DIFFERENT element - don't repeat
4. Look for modals/popups blocking the page

## Your Goal
{goal}

Keep explanations to ONE short sentence.
"""

    def get_prompt(self, state: State, goal: str) -> str:
        last_obs = state.get_last_observation()
        if not last_obs:
            return ""

        cur_axtree_txt = last_obs.axtree_txt
        cur_url = last_obs.url
        error_prefix = last_obs.error

        # Get history context (includes loop detection)
        history_context = state.view.get_prompt_context(max_events=10)

        prompt = f"""
# Current Browser State

## URL
{cur_url}

## Last Action Result
{self._format_last_result(state)}

## Action History
{history_context}

## Current Page Elements (Accessibility Tree)
Use these IDs (bid) for your actions:
{cur_axtree_txt}
""".strip()

        if state.intent:
            prompt += f"\n\n## Your Original Plan: {state.intent.approach}"

        # Add instructions
        prompt += """

## Instructions
- Pick at most THREE actions based on the accessibility tree above
- Use the exact bid from the tree
- fill() just types text - remember to click submit or press Enter after
- Only send_msg_to_user() when you have the COMPLETE answer
"""

        # Extra warning if stuck
        if state.is_stuck():
            print("=============== STUCK DETECTED ==============")
            
            # Get loop info for specific guidance
            loop_info = state.view.get_loop_info()
            
            prompt += f"""

🚨 YOU ARE STUCK - STOP REPEATING YOURSELF 🚨

You have been repeating actions that aren't working.
{f"Repeated pattern: {' → '.join(loop_info[0])}" if loop_info else ""}

MANDATORY: Try something COMPLETELY DIFFERENT:
1. Look at the accessibility tree - find a DIFFERENT element
2. Is there a modal/popup blocking? Look for "Close", "X", or "Dismiss"
3. Is this a date picker? Look for "Done" or "Apply" button
4. Did you fill a search box? Look for a search BUTTON to click
5. Scroll to find elements you haven't seen

DO NOT repeat any action you've already tried.
"""
        return prompt

    def _format_last_result(self, state: State) -> str:
        """Format the last action's result clearly."""
        last_action = state.get_last_action()
        last_obs = state.get_last_observation()
        
        if not last_action:
            return "No previous action (starting fresh)"
        
        result = f"Action: `{last_action.code}`\n"
        
        if last_obs:
            if last_obs.error:
                result += f"❌ FAILED: {last_obs.error}"
            elif last_obs.last_action_success:
                result += f"✓ SUCCESS"
                if "fill(" in last_action.code:
                    result += "\n⚠️ Text was typed but NOT submitted. Click a button or press Enter to submit."
            else:
                result += "⚠️ NO CHANGE: Action may not have worked"
        
        return result

    def _validate_action(self, action_code: str) -> Tuple[str, Optional[str]]:
        """
        Validate and fix common action syntax errors.
        Returns (fixed_action, error_message) or (original_action, None) if valid.
        """
        import re
        original = action_code
        error_msg = None
        
        # Fix fill() with 3 arguments (remove the third)
        # fill('123', 'text', true) -> fill('123', 'text')
        fill_match = re.match(r"fill\s*\(\s*(['\"][^'\"]+['\"])\s*,\s*(['\"][^'\"]*['\"])\s*,\s*\w+\s*\)", action_code)
        if fill_match:
            action_code = f"fill({fill_match.group(1)}, {fill_match.group(2)})"
            error_msg = f"Removed extra argument from fill(): {original} -> {action_code}"
        
        # Fix fill() with unquoted bid
        # fill(123, 'text') -> fill('123', 'text')
        fill_unquoted = re.match(r"fill\s*\(\s*(\d+)\s*,\s*(['\"].+['\"])\s*\)", action_code)
        if fill_unquoted:
            action_code = f"fill('{fill_unquoted.group(1)}', {fill_unquoted.group(2)})"
            error_msg = f"Added quotes to bid: {original} -> {action_code}"
        
        # Fix click() with unquoted bid
        # click(123) -> click('123')
        click_unquoted = re.match(r"click\s*\(\s*(\d+)\s*\)", action_code)
        if click_unquoted:
            action_code = f"click('{click_unquoted.group(1)}')"
            error_msg = f"Added quotes to bid: {original} -> {action_code}"
        
        # Fix press() with wrong quotes or format
        # press(Enter) -> press('Enter')
        press_unquoted = re.match(r"press\s*\(\s*(\w+)\s*\)", action_code)
        if press_unquoted and not action_code.startswith("press('") and not action_code.startswith('press("'):
            action_code = f"press('{press_unquoted.group(1)}')"
            error_msg = f"Added quotes to key: {original} -> {action_code}"
        
        return action_code, error_msg

    def reflect_on_failure(self, state: State, last_error: str) -> str:
        recent_history = state.view.get_prompt_context(max_events=5)
        
        response = self.client.beta.chat.completions.parse(
            model=self.llm,
            input=[{
                "role": "system",
                "content": """Analyze why the browser automation is stuck. Common issues:
1. Typed into a field but forgot to click submit or press Enter
2. Using wrong element ID
3. A modal/popup/datepicker is open and blocking
4. Need to click a "Done" or "Apply" button
5. Element not visible - need to scroll

Be specific about what to try differently."""
            }, {
                "role": "user", 
                "content": f"Goal: {state.goal}\n\nRecent Actions:\n{recent_history}\n\nLast error: {last_error}"
            }],
            text_format=ReflectionFormat
        )
        reflection = response.choices[0].message.parsed
        
        if self.speech:
            self.speech.speak(f"Hmm, that's not working. {reflection.new_approach}", wait=True)
        
        return reflection.new_approach

    def cleanup(self):
        """Shutdown the agent."""
        self._command_queue.put(None)
        self._agent_thread.join(timeout=5.0)