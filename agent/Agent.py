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


class OutputFormat(BaseModel):
    explanation: str
    code: str
    scratchpad: str

class IntentFormat(BaseModel):
    understanding: str
    approach: str

class ChangedIntentFormat(BaseModel):
    changes: str
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
            multiaction=False
        )
        self.obs = None
        self.speech_thread = None
        
        self._command_queue = queue.Queue()
        self._stop_requested = False
        self._is_running = False
        self._lock = threading.Lock()
        
        self._agent_thread = threading.Thread(target=self.agent_loop, daemon=True)
        self._agent_thread.start()

    def agent_loop(self):
        """Main loop running on dedicated thread - browser lives here."""
        self.env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank'},
            headless=False,
            tags_to_mark='standard_html'
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

                self.execute_goal(command.goal, command.on_complete)

            except Exception as e:
                print(f"❌ Agent loop error: {e}")

        if self.env:
            self.env.close()
    
    def get_intent(self, goal):
        response = self.client.responses.parse(
            model=self.llm,
            input=[{
                "role": "system", 
                "content": """You are a Browser Agent with full access to a browser about to help with a web task. Briefly describe your understanding and approach. Don't add extra functionality, do what the user desires.

                    Be conversational. Do NOT ask the user for permission - just complete the task."""
            }, {
                "role": "user", 
                "content": f"Goal: {goal}"
            }],
            text_format=IntentFormat,
        )
        return response.output_parsed
    
    def rewrite_plan(self, new_information: str):
        state = self.current_state
        history_str = state.view.get_prompt_context(max_events=10)
        current_approach = state.intent.approach if state.intent else "Starting fresh"
        
        response = self.client.responses.parse(
            model=self.llm,
            input=[{
                "role": "system", 
                "content": f"""
                You are a Browser Automation Agent. 
                You are in the middle of a task, but the user has issued a INTERRUPTION with new instructions.

                # Current Context
                **Original Goal:** "{state.goal}"
                **Current Approach:** "{current_approach}"
                
                # Recent Action History (What you have already done)
                {history_str}

                # Your Task
                The user has provided NEW INFORMATION. You must update your 'Understanding' and 'Approach'.
                
                1. **Compare**: How does the new info change the original goal?
                2. **Evaluate History**: Look at the history. 
                   - If the new info contradicts what you've already done, your new approach must explicitly say to go back or fix it.
                   - If the new info is just an add-on (e.g. "also make it red"), append it to the current plan without restarting.
                3. **Output**: A conversational summary of the NEW plan.

                # Output Rules
                - **changes**: Explain what the changes are from the new information as if you are explaining it to someone, keep it to only ONE sentence
                - **approach**: A step-by-step plan focusing on what to do *next*.
                - Be concise and direct.
                """
            }, {
                "role": "user", 
                "content": f"New Information/Correction: {new_information}"
            }],
            text_format=ChangedIntentFormat,
        )
        return response.output_parsed
    
    def add_goal(self, new_goal_text: str, max_steps: int):
        """
        Dynamically adds information to the current goal if running,
        or starts a new run if idle.
        """
        with self._lock:
            if self._is_running and self.current_state:
                print(f"UPDATE: Adding to existing goal: '{new_goal_text}'")
                
                
                self.current_state.add_note(f"USER NOTE: {new_goal_text}")
                new_intent = self.rewrite_plan(new_goal_text)
                self.current_state.intent = new_intent.approach
                if self.speech:                    
                    self.speech.speak(
                        f"{new_intent.changes}", 
                        wait=True, 
                        ignore_mute=True
                    )
                    
                    self.speech.unmute()
                    self.max_steps += max_steps

                return
        
        print("\n UPDATE: Agent idle, starting new goal.")
        self.run(new_goal_text, max_steps)
    def execute_goal(self, goal: str, on_complete: Optional[callable] = None, max_steps: int = 50):
        """Execute a goal - runs on agent thread."""
        with self._lock:
            self._stop_requested = False
            self._is_running = True
        self.max_steps = max_steps
        self.current_state = State(goal=goal)
        intent = self.get_intent(goal)
        self.current_state.intent = intent
        
        system_message = self.get_system_message(goal)

        if self.speech:
            self.speech.speak(intent.approach, wait=True)

        if self.obs:
            initial_obs_event = self.current_state.update_from_observation(self.obs)
            self.current_state.add_event(initial_obs_event)
            
        try:
            for step_num in range(self.max_steps):
                with self._lock:
                    if self._stop_requested:
                        print("\n🛑 Task interrupted by new voice command.")
                        if self.speech:
                            self.speech.stop()
                        return

                print(f"\n--- Step {step_num + 1}/{self.max_steps} ---")

                if self.current_state.consecutive_errors > 5:
                    if self.speech:
                        self.speech.speak("I'm having too much trouble. I'll stop here.", wait=True)
                    break

                try:
                    done = self.step(system_message)
                    
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
                    self.current_state.add_event(error_event)
                    
                    try:
                        self.obs, _ = self.env.reset()
                        reset_event = self.current_state.update_from_observation(self.obs)
                        self.current_state.add_event(reset_event)
                    except Exception as reset_error:
                        print(f"❌ Could not recover: {reset_error}")
                        if self.speech:
                            self.speech.speak("I cannot recover the browser.", wait=True)
                        break

            print(f"\n⏰ Reached max steps ({self.max_steps})")
            if self.speech:
                self.speech.speak("I reached the maximum number of steps without finishing.", wait=True)
            
        finally:
            with self._lock:
                self._is_running = False
            if on_complete:
                on_complete(False)
            print("\n🎤 Ready for next voice command. Press Option to speak...")

    def set_speech_thread(self, speech_thread):
        self.speech_thread = speech_thread

    def step(self, system_message: str) -> bool:
        """Execute a single step with speech. Returns True if task is complete."""
        
        prompt = self.get_prompt(self.current_state)

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
        new_notes = event_data.scratchpad

        print(f"Thought: {explanation}")
        print(f"Action: {action_code}")

        action_code, validation_error = self.validate_action(action_code)
        if validation_error:
            print(f"⚠️ Fixed action syntax: {validation_error}")

        action_event = BrowserAction(
            source=EventSource.AGENT,
            thought=explanation,
            code=action_code
        )
        self.current_state.add_event(action_event)
        if new_notes:
            self.current_state.add_note(new_notes)

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
                self.current_state.add_event(error_event)
                return False
            elif self.speech:
                response = self.compile_result(action_code)
                self.speech.speak(f"Here is what I found: {response}", wait = True)
                return True
            
        if self.speech:
            if self.speech_thread and self.speech_thread.is_alive():
                print(f"Skipping speech (Previous still active): '{explanation}'")
            else:
                self.speech_thread = self.speech.speak(explanation, wait = False)

        try:
            self.obs, reward, done, truncated, info = self.env.step(action_code)

            obs_event = self.current_state.update_from_observation(self.obs)
            self.current_state.add_event(obs_event)

            if obs_event.error:
                print(f"⚠️ Action error: {obs_event.error}")
            elif not obs_event.last_action_success:
                 print("⚠️ No page change detected")
            else:
                 print("✅ Page updated")
            

            return False

        except Exception as e:
            if self.speech_thread and self.speech_thread.is_alive():
                self.speech_thread.join()
            print(f"❌ Environment error: {e}")
            raise e

    def run(self, goal: str, max_steps = 50, on_complete: Optional[callable] = None):
        """Queue a goal to be executed (thread-safe)."""
        if self.speech:
            self.speech.stop()

        while not self._command_queue.empty():
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                break

        self._command_queue.put(AgentCommand(goal=goal, on_complete=on_complete))


    def compile_result(self, message: str) -> str:
        """Compile the agent's findings into a concise spoken response."""
        
        goal = self.current_state.goal if self.current_state else "the requested task"
        notes = self.current_state.scratchpad if self.current_state else ""
        
        system_prompt = f"""\
    You are summarizing results for a voice assistant. The user asked: "{goal}"

    A browser agent completed this task and gathered the following information. Your job is to deliver the findings as natural speech.

    Rules:
    - Be conversational and direct—no filler phrases like "I found that" or "Here's what I discovered"
    - Lead with the most important information first
    - Keep it under 3 sentences unless the data requires more
    - Use natural spoken phrasing (say "around 300 dollars" not "$299.99")
    - Don't list raw URLs or technical details, URLS should just be on the current screen in the browser
    - Reference any notes the agent recorded: {notes if notes else "None"}

    Respond with ONLY the spoken text—no labels, no formatting. Don't ask any questions if there are any, just the information found only."""

        response = self.client.responses.create(
            model=self.llm,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Agent's raw findings:\n{message}"}
            ],
        )
        print(response.output_text)
        return response.output_text
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
            press('bid', 'key_comb')       # Press a key (after fill to submit)
            goto('https://url.com')    # Navigate to URL
            scroll(0, 500)             # Scroll down (positive y = down)
            send_msg_to_user('msg')    # Return final answer to user
            ```

            ## SYNTAX RULES - IMPORTANT
            - bid and text must be in QUOTES: fill('123', 'hello') ✓
            - fill() takes EXACTLY 2 arguments: fill('bid', 'text') ✓
            - DO NOT add extra parameters: fill('123', 'hello', true) ✗ WRONG
            - scroll() takes 2 numbers (no quotes): scroll(0, 500) ✓

            ## OUTPUT FORMAT
            You must respond with this exact structure:
            - **explanation**: A short, one-sentence explaining your action as if you are speaking to someone.
            - **code**: The specific action syntax (e.g., `click('55')`).
            - **scratchpad**: THIS IS YOUR MEMORY. Anything you write here gets SAVED to your permanent scratchpad. 
                - USE THIS TO SAVE IMPORTANT DATA YOU FIND THAT IS RELEVANT TO THE PLAN.
                - If you have no new data to save, leave this empty string "".
                - DO NOT repeat previous items on your scratchpad; just add NEW findings. THIS IS INCREDIBLY IMPORTANT

            ## Workflow Patterns
            **Search:** fill('searchbox_id', 'query') → click('search_btn_id')
            **Form:** fill('field1', 'value1') → fill('field2', 'value2') → click('submit_btn')

            ## CRITICAL RULES
            1. REFERENCE YOUR SCRATCHPAD AND PLAN FIRST ALWAYS BEFORE TAKING ACTION
            2. Use exact bid values from the accessibility tree
            3. If action fails, try a DIFFERENT element - don't repeat
            4. Look for modals/popups blocking the page
            5. fill() only types text - you MUST click a button or press Enter to submit

            ## Your Goal
            {goal}

            ## Your plan
            {self.current_state.intent.approach}
            # IMPORTANT - You must fulfill all tasks outlined in your plan.
            """

    def get_prompt(self, state: State) -> str:
        last_obs = state.get_last_observation()
        if not last_obs:
            return ""

        cur_axtree_txt = last_obs.axtree_txt
        cur_url = last_obs.url

        history_context = state.view.get_prompt_context(max_events=5)

        prompt = f"""
            # Current Browser State

            ## URL
            {cur_url}

            ## Action History
            {history_context}

            ## Current Page Elements (Accessibility Tree)
            {cur_axtree_txt}
            """.strip()

        if state.is_stuck():
            loop_info = state.view.get_loop_info()
            stuck_prompt = self.get_stuck_prompt(loop_info)
            prompt += stuck_prompt
        return prompt

    def get_stuck_prompt(self, loop_info):
        return f"""

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

    def validate_action(self, action_code: str) -> Tuple[str, Optional[str]]:
        """
        Validate and fix common action syntax errors.
        Returns (fixed_action, error_message) or (original_action, None) if valid.
        """
        import re
        original = action_code
        error_msg = None
        
        fill_match = re.match(r"fill\s*\(\s*(['\"][^'\"]+['\"])\s*,\s*(['\"][^'\"]*['\"])\s*,\s*\w+\s*\)", action_code)
        if fill_match:
            action_code = f"fill({fill_match.group(1)}, {fill_match.group(2)})"
            error_msg = f"Removed extra argument from fill(): {original} -> {action_code}"

        fill_unquoted = re.match(r"fill\s*\(\s*(\d+)\s*,\s*(['\"].+['\"])\s*\)", action_code)
        if fill_unquoted:
            action_code = f"fill('{fill_unquoted.group(1)}', {fill_unquoted.group(2)})"
            error_msg = f"Added quotes to bid: {original} -> {action_code}"
        
        click_unquoted = re.match(r"click\s*\(\s*(\d+)\s*\)", action_code)
        if click_unquoted:
            action_code = f"click('{click_unquoted.group(1)}')"
            error_msg = f"Added quotes to bid: {original} -> {action_code}"
        
        
        return action_code, error_msg

    def cleanup(self):
        self._command_queue.put(None)
        self._agent_thread.join(timeout=5.0)