


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


class IntentFormat(BaseModel):
    understanding: str  # What the user actually wants
    approach: str       # High-level strategy (1-2 sentences)

class OutputFormat(BaseModel):
    explanation: str
    code: str


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


class State:
    """Tracks the current state of the environment."""
    def __init__(self, goal: str):
        self.goal = goal
        self.obs = None
        self.actions = []
        self.consecutive_errors = 0
        self.intent = None

    def set_obs(self, obs: dict):
        self.obs = obs

    def get_obs(self) -> dict:
        return self.obs

    def get_actions(self):
        return self.actions

    def add_action(self, action: str):
        self.actions.append(action)

    def record_success(self):
        self.consecutive_errors = 0

    def record_error(self):
        self.consecutive_errors += 1

    def get_errors(self):
        return self.consecutive_errors
    def set_intent(self, intent: IntentFormat):
        self.intent = intent
    def get_intent(self):
        return self.intent
