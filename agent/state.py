


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
import hashlib


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
        self.consecutive_no_change = 0  # NEW
        self.last_page_hash = None      # NEW
        self.last_url = None            # NEW
    
    def compute_page_hash(self, obs: dict) -> str:
        """Create a hash of the meaningful page state."""
        # Combine URL + accessibility tree structure
        url = obs.get('url', '')
        axtree = str(obs.get('axtree_object', ''))[:5000]  # First 5k chars
        
        content = f"{url}|{axtree}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def check_page_changed(self, new_obs: dict) -> bool:
        """Returns True if page actually changed."""
        new_hash = self.compute_page_hash(new_obs)
        new_url = new_obs.get('url', '')
        
        changed = (
            new_hash != self.last_page_hash or 
            new_url != self.last_url
        )
        
        # Update tracking
        self.last_page_hash = new_hash
        self.last_url = new_url
        
        return changed
    
    def record_no_change(self):
        self.consecutive_no_change += 1
        self.consecutive_errors += 1  # Treat as error too
    def record_change(self):
        self.consecutive_no_change = 0

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
    
    def is_stuck(self):
        return self.consecutive_no_change >= 3 or self.consecutive_errors >= 3
