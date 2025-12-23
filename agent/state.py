


import queue
import threading
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


from openai import OpenAI
import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
from pydantic import BaseModel
from agent.SpeechOutput import SpeechOutput
import hashlib
from agent.Action import *


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


# class State:
#     """Tracks the current state of the environment."""
#     def __init__(self, goal: str):
#         self.goal = goal
#         self.obs = None
#         self.actions = []
#         self.consecutive_errors = 0
#         self.intent = None
#         self.consecutive_no_change = 0  # NEW
#         self.last_page_hash = None      # NEW
#         self.last_url = None            # NEW
    
#     def compute_page_hash(self, obs: dict) -> str:
#         """Create a hash of the meaningful page state."""
#         # Combine URL + accessibility tree structure
#         url = obs.get('url', '')
#         axtree = str(obs.get('axtree_object', ''))[:5000]  # First 5k chars
        
#         content = f"{url}|{axtree}"
#         return hashlib.md5(content.encode()).hexdigest()
    
#     def check_page_changed(self, new_obs: dict) -> bool:
#         """Returns True if page actually changed."""
#         new_hash = self.compute_page_hash(new_obs)
#         new_url = new_obs.get('url', '')
        
#         changed = (
#             new_hash != self.last_page_hash or 
#             new_url != self.last_url
#         )
        
#         # Update tracking
#         self.last_page_hash = new_hash
#         self.last_url = new_url
        
#         return changed
    
#     def record_no_change(self):
#         self.consecutive_no_change += 1
#         self.consecutive_errors += 1  # Treat as error too
#     def record_change(self):
#         self.consecutive_no_change = 0

#     def set_obs(self, obs: dict):
#         self.obs = obs

#     def get_obs(self) -> dict:
#         return self.obs

#     def get_actions(self):
#         return self.actions

#     def add_action(self, action: str):
#         self.actions.append(action)

#     def record_success(self):
#         self.consecutive_errors = 0

#     def record_error(self):
#         self.consecutive_errors += 1

#     def get_errors(self):
#         return self.consecutive_errors
#     def set_intent(self, intent: IntentFormat):
#         self.intent = intent
#     def get_intent(self):
#         return self.intent
    
#     def is_stuck(self):
#         return self.consecutive_no_change >= 3 or self.consecutive_errors >= 3
from pydantic import BaseModel

class IntentFormat(BaseModel):
    understanding: str
    approach: str

@dataclass
class State:
    # --- OpenHands Core Structure ---
    history: List[Event] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # --- Your Custom Tracking Logic ---
    goal: str = ""
    intent: Optional[IntentFormat] = None
    
    # Tracking metrics
    consecutive_errors: int = 0
    consecutive_no_change: int = 0
    
    # Hashing for change detection
    _last_page_hash: Optional[str] = None
    _last_url: Optional[str] = None
    _view: Optional[View] = None
    _history_checksum: int = -1

    @property
    def view(self) -> View:
        """
        OpenHands style view caching. 
        Recreates view only if history length changed.
        """
        current_checksum = len(self.history)
        if current_checksum != self._history_checksum:
            self._view = View(self.history)
            self._history_checksum = current_checksum
        return self._view

    def add_event(self, event: Event):
        self.history.append(event)

    def get_last_observation(self) -> Optional[BrowserObservation]:
        for event in reversed(self.history):
            if isinstance(event, BrowserObservation):
                return event
        return None

    def get_last_action(self) -> Optional[BrowserAction]:
        for event in reversed(self.history):
            if isinstance(event, BrowserAction):
                return event
        return None

    # --- Your Logic Ported to use Events ---

    def compute_page_hash(self, obs_dict: dict) -> str:
        url = obs_dict.get('url', '')
        axtree = str(obs_dict.get('axtree_object', ''))[:5000]
        content = f"{url}|{axtree}"
        return hashlib.md5(content.encode()).hexdigest()

    # In state.py - update_from_observation()

    def update_from_observation(self, obs_dict: dict) -> BrowserObservation:
        """
        Takes raw gym dict, updates state metrics, returns typed Event.
        """
        # 1. Calculate Hash & Change Detection
        new_hash = self.compute_page_hash(obs_dict)
        new_url = obs_dict.get('url', '')
        
        has_changed = (
            new_hash != self._last_page_hash or 
            new_url != self._last_url
        )
        
        # 2. Update Internal Metrics
        error_msg = obs_dict.get('last_action_error', '')
        
        # Determine success status
        action_success = True
        
        if error_msg:
            self.consecutive_errors += 1
            action_success = False
        elif not has_changed:
            self.consecutive_no_change += 1
            self.consecutive_errors += 1  # Treat as error
            action_success = False
            error_msg = "Action had no visible effect on the page"  # ← ADD THIS
        else:
            # Success reset
            self.consecutive_errors = 0
            self.consecutive_no_change = 0
            self._last_page_hash = new_hash  # Only updated here!
            self._last_url = new_url
        
        # Always update hash tracking (so we detect NEW no-changes)
        self._last_page_hash = new_hash
        self._last_url = new_url

        # 3. Create and Return the Event
        return BrowserObservation(
            source=EventSource.ENVIRONMENT,
            url=new_url,
            axtree_txt=flatten_axtree_to_str(obs_dict.get('axtree_object', '')),
            error=error_msg,
            last_action_success=action_success
        )

    def is_stuck(self):
        return self.consecutive_no_change >= 3 or self.consecutive_errors >= 3