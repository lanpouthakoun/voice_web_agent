


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

class IntentFormat(BaseModel):
    understanding: str
    approach: str

@dataclass
class State:
    history: List[Event] = field(default_factory=list)
    inputs: Dict[str, Any] = field(default_factory=dict)
    scratchpad: str = ""
    
    goal: str = ""
    intent: Optional[IntentFormat] = None
    
    consecutive_errors: int = 0
    consecutive_no_change: int = 0
    
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
            self._view = View(self.history, self.scratchpad)
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

    def add_note(self, note):
        self.scratchpad += f"\n {note}"
    
    def get_scratchpad(self):
        return self.scratchpad


    def compute_page_hash(self, obs_dict: dict) -> str:
        url = obs_dict.get('url', '')
        axtree = str(obs_dict.get('axtree_object', ''))[:5000]
        content = f"{url}|{axtree}"
        return hashlib.md5(content.encode()).hexdigest()

    def update_from_observation(self, obs_dict: dict) -> BrowserObservation:
        """
        Takes raw gym dict, updates state metrics, returns typed Event.
        """
        new_hash = self.compute_page_hash(obs_dict)
        new_url = obs_dict.get('url', '')
        focused_bid = obs_dict.get('focused_element_bid', '')
        
        has_changed = (
            new_hash != self._last_page_hash or 
            new_url != self._last_url
        )
        
        error_msg = obs_dict.get('last_action_error', '')
        
        action_success = True
        
        if error_msg:
            self.consecutive_errors += 1
            action_success = False
        elif not has_changed:
            self.consecutive_no_change += 1
            self.consecutive_errors += 1 
            action_success = False
            error_msg = "Action had no visible effect on the page" 
        else:
            self.consecutive_errors = 0
            self.consecutive_no_change = 0
            self._last_page_hash = new_hash  
            self._last_url = new_url
        
        self._last_page_hash = new_hash
        self._last_url = new_url

        return BrowserObservation(
            source=EventSource.ENVIRONMENT,
            url=new_url,
            axtree_txt=flatten_axtree_to_str(obs_dict.get('axtree_object', '')),
            error=error_msg,
            last_action_success=action_success,
            focused_element_bid = focused_bid
        )

    def is_stuck(self):
        return self.consecutive_no_change >= 3 or self.consecutive_errors >= 3