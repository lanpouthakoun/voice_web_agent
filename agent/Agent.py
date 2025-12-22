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

Make sure you have an explanation and an action.
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

@dataclass
class Plan:
    """A high-level plan for accomplishing the goal."""
    steps: list[str]
    current_step: int = 0
    
    def current(self) -> str:
        if self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return "All planned steps completed"
    
    def advance(self):
        self.current_step += 1
    
    def is_complete(self) -> bool:
        return self.current_step >= len(self.steps)
    
    def format_for_prompt(self) -> str:
        lines = []
        for i, step in enumerate(self.steps):
            if i < self.current_step:
                marker = "✓"
            elif i == self.current_step:
                marker = "→"
            else:
                marker = " "
            lines.append(f"{marker} {i+1}. {step}")
        return '\n'.join(lines)
    
class State:
    """Tracks the current state of the environment. No plan is needed, just keeps track of whether the goal is accomplished or not."""
    def __init__(self, goal: str, llm: str, client: OpenAI):
        self.complete = False
        self.goal = goal
        self.llm = llm
        self.obs = None
        self.client = client
    def set_obs(self, obs: dict):
        self.obs = obs
    def get_obs(self) -> dict:
        return self.obs
    
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
    def is_complete(self) -> bool:
        """Returns True if the goal is accomplished, False otherwise. Uses AI to check if the goal is accomplished"""
        if self.obs is None:
            return False
        response = self.client.chat.completions.create(
            model=self.llm,
            messages=[{"role": "user", "content": self.get_prompt()}],
        )
        text = response.choices[0].message.content
        return text == "true"

class Agent:
    def __init__(self, llm: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.llm = llm
        self.api_key = api_key
        self.env = gym.make('browsergym/openended', task_kwargs={'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'}, headless=False, tags_to_mark='all')
        self.action_space = HighLevelActionSet(subsets=['chat', 'nav', 'bid'], strict=False, multiaction=False)

        self.obs, _ = self.env.reset()

    def run(self, goal: str):
        """"
        Args:
            goal: The goal the user wants to accomplish
        Returns:
            None: this should be a loop that runs until the goal is accomplished or the actions break
        """
        state = State(goal=goal, llm=self.llm, client=self.client)
        system_message = self.get_system_message(goal)
        while not state.is_complete():
            state = self.step(state, goal, system_message)

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
        action = event.code
        self.obs, *_ = self.env.step(action)
        state.set_obs(self.obs)
        return state
    
    def get_system_message(self, goal: str) -> str:
        return f"""\
        # Instructions
        Review the current state of the page and all other information to find the best
        possible next action to accomplish your goal. Your answer will be interpreted
        and executed by a program, make sure to follow the formatting instructions.

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
        prompt = f"""
            # Current Page URL:
            {cur_url}

            # Current Accessibility Tree:
            {cur_axtree_txt}

            Here is an example with chain of thought of a valid action when clicking on a button:
            "
            In order to accomplish my goal I need to click on the button with bid 12
            ```click("12")```
            "
            """.strip()
        prompt += CONCISE_INSTRUCTION
        return prompt

if __name__ == "__main__":
    load_dotenv('./.env', override=True)
    agent = Agent(llm="gpt-5.2", api_key=os.getenv('OPENAI_API_KEY'))
    agent.run("Open a new tab and navigate to youtube.com. Search up the latest news in the world.")