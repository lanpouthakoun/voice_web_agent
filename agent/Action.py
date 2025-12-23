from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from collections import Counter


class EventSource(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENVIRONMENT = "environment"


@dataclass
class Event:
    timestamp: datetime = field(default_factory=datetime.now)
    source: EventSource = EventSource.AGENT


@dataclass
class Action(Event):
    """Base class for things the agent does"""
    thought: str = ""


@dataclass
class BrowserAction(Action):
    """Specific action to interact with the browser"""
    code: str = ""


@dataclass
class MessageAction(Action):
    """Action to speak to the user"""
    content: str = ""


@dataclass
class Observation(Event):
    """Base class for things the agent sees"""
    content: str = ""


@dataclass
class BrowserObservation(Observation):
    """Specific observation from BrowserGym"""
    url: str = ""
    axtree_txt: str = ""
    error: str = ""
    focused_element_bid: str = ""

    last_action_success: bool = True
    
    @property
    def clean_content(self):
        return f"URL: {self.url}\nError: {self.error}\nTree: {self.axtree_txt[:1000]}..."


class View:
    """
    Renders event history into prompts for the LLM.
    """
    
    def __init__(self, history: List[Event]):
        self.history = history

    def get_prompt_context(self, max_events: int = 10) -> str:
        """
        Renders history in a strict Thought -> Action -> Observation format.
        """
        if not self.history:
            return "No previous actions. Initial State: Unknown."
        
        lines = []
        
        current_state = self.get_current_state_description()
        lines.append("## CURRENT BROWSER STATE")
        lines.append(current_state)
        lines.append("")
        
        loop_warning = self.detect_loops()
        if loop_warning:
            lines.append(loop_warning)
            lines.append("")

        lines.append("## EXECUTION HISTORY")
        recent_context = self.format_trajectory(max_events)
        lines.append(recent_context)
        
        return "\n".join(lines)

    def get_current_state_description(self) -> str:
        """Finds the most recent BrowserObservation to render the current 'screen'."""
        for event in reversed(self.history):
            if isinstance(event, BrowserObservation):
                return (
                    f"URL: {event.url}\n"
                    f"Active Accessibility Tree:\n"
                    f"{event.axtree_txt}..." 
                )
        return "Browser not yet initialized."

    def format_trajectory(self, max_events: int) -> str:
        """
        Formats events as:
        [Step N]
        Thought: ...
        Action: ...
        Observation: ...
        """
        context_lines = []
        
        recent_events = self.history[-max_events:]
        
        step_count = 1
        
        for event in recent_events:
            if isinstance(event, MessageAction):
                context_lines.append(f"🤖 [AGENT MESSAGE]: {event.content}")
                context_lines.append("---")

            elif isinstance(event, BrowserAction):
                block = f"step_{step_count}\n"
                if event.thought:
                    block += f"💭 THOUGHT: {event.thought}\n"
                block += f"🛠️ ACTION: {event.code}"
                context_lines.append(block)
                step_count += 1

            elif isinstance(event, BrowserObservation):
                status = "SUCCESS" if event.last_action_success else "FAIL"
                if event.error:
                    content = f"❌ ERROR: {event.error}"
                else:
                    content = f"URL: {event.url}\n Focused Element Bid: {event.focused_element_bid} \n Active Accessible Tree {event.axtree_txt}"
                
                context_lines.append(f"👀 OBSERVATION [{status}]:\n{content}")
                context_lines.append("---")

            elif isinstance(event, Observation):
                context_lines.append(f"👀 OBSERVATION: {event.content}")
                context_lines.append("---")
        
        return "\n".join(context_lines)

    def detect_loops(self) -> Optional[str]:
        """Detect if agent is stuck in a loop and return a warning."""
        actions = self.get_all_actions()
        if len(actions) < 4: return None
        
        for pattern_len in [2, 3]:
            if len(actions) >= pattern_len * 2:
                recent = actions[-pattern_len:]
                previous = actions[-pattern_len*2:-pattern_len]
                if recent == previous:
                    return f"⚠️ SYSTEM ALERT: Loop detected. You performed {' -> '.join(recent)} twice. STOP and try a different strategy."
        
        if len(actions) >= 5:
            if actions[-1] == actions[-2] == actions[-3]:
                 return f"⚠️ SYSTEM ALERT: You are repeating `{actions[-1]}`. It is not working. Do not do this again."
        return None

    def get_all_actions(self) -> List[str]:
        return [e.code for e in self.history if isinstance(e, BrowserAction)]
        
    def get_last_n_actions(self, n: int = 3) -> List[str]:
        actions = self.get_all_actions()
        return actions[-n:] if len(actions) >= n else actions

    def is_repeating_actions(self) -> bool:
        return self.detect_loops() is not None
    def get_loop_info(self) -> Optional[Tuple[List[str], int]]:
        """Returns (pattern, repetition_count) if a loop is detected."""
        actions = self.get_all_actions()
        
        for pattern_len in [2, 3, 4]:
            if len(actions) >= pattern_len * 2:
                pattern = actions[-pattern_len:]
                count = 1
                pos = len(actions) - pattern_len * 2
                while pos >= 0:
                    if actions[pos:pos+pattern_len] == pattern:
                        count += 1
                        pos -= pattern_len
                    else:
                        break
                if count >= 2:
                    return (pattern, count)
        
        return None