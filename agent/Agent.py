import gymnasium as gym
import browsergym.core
import os
import re
from dotenv import load_dotenv
from browsergym.utils.obs import flatten_axtree_to_str
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional
from browsergym.core.action.highlevel import HighLevelActionSet
from pydantic import BaseModel


class OutputFormat(BaseModel):
    explanation: str
    code: str

CONCISE_INSTRUCTION = """\

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
```send_msg_to_user("$279.49")```
"

Make sure you have an explanation and an action. Make the explanation such that it sounds like you are telling this to someone.
"""


@dataclass
class ActionResult:
    action: str
    thought: str
    url_before: str
    url_after: str
    success: bool
    error: str | None = None

@dataclass 
class AgentMemory:
    """Tracks action history with outcomes."""
    history: List[ActionResult] = field(default_factory=list)
    max_entries: int = 15
    
    def add(self, result: ActionResult):
        self.history.append(result)
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
    
    def format_for_prompt(self) -> str:
        if not self.history:
            return "No previous actions."
        
        lines = []
        for i, h in enumerate(self.history[-7:], 1):
            status = "✓" if h.success else "✗"
            lines.append(f"{i}. [{status}] {h.action}")
            if h.url_before != h.url_after:
                lines.append(f"   → Navigated to: {h.url_after}")
            if h.error:
                lines.append(f"   Error: {h.error}")
            if h.thought:
                lines.append(f"   Reasoning: {h.thought[:100]}...")
        return '\n'.join(lines)
    
    def get_error_count(self, last_n: int = 3) -> int:
        """Count recent consecutive errors."""
        count = 0
        for h in reversed(self.history[-last_n:]):
            if not h.success:
                count += 1
            else:
                break
        return count

    
class State:
    """Tracks the current state of the environment. No plan is needed, just keeps track of whether the goal is accomplished or not."""
    def __init__(self, goal: str, llm: str, client: OpenAI):
        self.complete = False
        self.goal = goal
        self.llm = llm
        self.obs = None
        self.client = client
        self.actions = []
        self.consecutive_errors = 0

    def set_obs(self, obs: dict):
        self.obs = obs
    def get_obs(self) -> dict:
        return self.obs
    
    def get_actions(self):
        return self.actions
    
    def get_prompt(self) -> str:
        return f"""
        # Goal:
        {self.goal}

        # Current Accessibility Tree:
        {flatten_axtree_to_str(self.obs['axtree_object'])}

        # Current Page URL:
        {self.obs['url']}

        Return true if the goal is accomplished, false otherwise.
        """
    def add_action(self, action):
        self.actions.append(action)
    def is_complete(self) -> bool:
        if self.obs is None:
            return False
        response = self.client.chat.completions.create(
            model=self.llm,
            messages=[{"role": "user", "content": self.get_prompt()}],
        )
        text = response.choices[0].message.content.strip().lower()
        return "true" in text or "yes" in text or "accomplished" in text
    
    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1
    def get_errors(self):
        return self.consecutive_errors



class BrowserAgent:
    def __init__(self, llm: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.llm = llm
        self.api_key = api_key
        self.env = gym.make('browsergym/openended', task_kwargs={'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'},headless=False, tags_to_mark='all')
        self.action_space = HighLevelActionSet(subsets=['chat', 'nav', 'bid'], strict=False, multiaction=True)
        self.obs, _ = self.env.reset()
        self.last_action = None

    def run(self, goal: str, max_steps: int = 50):
        state = State(goal=goal, llm=self.llm, client=self.client)
        system_message = self.get_system_message(goal)
        
        try:
            for step_num in range(max_steps):
                print(f"\n--- Step {step_num + 1}/{max_steps} ---")
                
                if state.is_complete():
                    print("Goal accomplished!")
                    return True
                
                # Stop if stuck
                if state.get_errors() > 5:
                    print("Too many repeated actions, stopping.")
                    return False
                    
                state = self.step(state, goal, system_message)
                if state == True:
                    break
            
            print(f"Reached max steps ({max_steps})")
            return False
            
        finally:
            self.env.close()

    def step(self, state: State, goal: str, system_message: str):
        """
        Args:
            state: The current state of the environment
            goal: The overall goal the user wants to accomplish
        Returns:
            State: The new state of the environment
        """
        prompt = self.get_prompt(state, goal)
        response = self.client.responses.parse(
            model=self.llm,
            input=[{"role": "system", "content": system_message}, {"role": "user", "content": prompt}],
            text_format=OutputFormat,
        )
        event = response.output_parsed
        explanation = event.explanation
        print(explanation)
        action = event.code
        state.add_action(action)
        
        self.obs, *_ = self.env.step(action)
        if self.last_action == action:
            state.record_error()
        else:
            state.record_success()

        self.last_action = action

        if "send_msg_to_user" in action:
            print(action)
            print("Message sent to user - task complete")
            # text_to_speech(final)
        

        state.set_obs(self.obs)
        return state
    
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
        error_prefix = obs['last_action_error']
        prompt = f"""
            {error_prefix}
            # Current Page URL:
            {cur_url}

            # Current Accessibility Tree:
            {cur_axtree_txt}

             # Previous Actions:
            {' '.join(state.get_actions())}

            Here is an example with chain of thought of a valid action when clicking on a button:
            "
            In order to accomplish my goal I need to click on the button with bid 12
            ```click("12")```
            "
            """.strip()
        prompt += CONCISE_INSTRUCTION
        if state.get_errors() > 3:
            prompt += "You are failing this task with your standard approach. Try a new approach."
        return prompt

if __name__ == "__main__":
    load_dotenv('./.env', override=True)
    agent = BrowserAgent(llm="gpt-5.2", api_key=os.getenv('OPENAI_API_KEY'))
    agent.run("I will arrive Pittsburgh Airport soon. Provide the name of a Hilton hotel in the vicinity, if available. Then, tell me the the walking distance to the nearest supermarket own by a local company from the hotel.")