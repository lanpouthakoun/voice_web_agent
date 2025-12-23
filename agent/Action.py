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
    last_action_success: bool = True
    
    @property
    def clean_content(self):
        return f"URL: {self.url}\nError: {self.error}\nTree: {self.axtree_txt[:1000]}..."


class View:
    """
    Renders event history into prompts for the LLM.
    Refactored to mimic OpenHands/ReAct style: State + Trajectory (Thought/Action/Observation).
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
        
        # 1. CURRENT STATE (Crucial: OpenHands always puts current state first or very clearly)
        # We find the last observation to show the current "Screen"
        current_state = self._get_current_state_description()
        lines.append("## CURRENT BROWSER STATE")
        lines.append(current_state)
        lines.append("")
        
        # 2. LOOP DETECTION (Keep this, it's a great feature)
        loop_warning = self._detect_loops()
        if loop_warning:
            lines.append(loop_warning)
            lines.append("")

        # 3. TRAJECTORY (History)
        # We render the last N events to show the "Story" of execution
        lines.append("## EXECUTION HISTORY")
        recent_context = self._format_trajectory(max_events)
        lines.append(recent_context)
        
        return "\n".join(lines)

    def _get_current_state_description(self) -> str:
        """Finds the most recent BrowserObservation to render the current 'screen'."""
        # Search backwards for the last browser observation
        for event in reversed(self.history):
            if isinstance(event, BrowserObservation):
                return (
                    f"URL: {event.url}\n"
                    f"Active Accessibility Tree:\n"
                    f"{event.axtree_txt[:2000]}..." # Give it plenty of token space
                )
        return "Browser not yet initialized."

    def _format_trajectory(self, max_events: int) -> str:
        """
        Formats events as:
        [Step N]
        Thought: ...
        Action: ...
        Observation: ...
        """
        context_lines = []
        
        # Get last N events, but ensure we don't slice in the middle of a thought/action pair if possible
        recent_events = self.history[-max_events:]
        
        step_count = 1
        
        for event in recent_events:
            if isinstance(event, MessageAction):
                context_lines.append(f"🤖 [AGENT MESSAGE]: {event.content}")
                context_lines.append("---")

            elif isinstance(event, BrowserAction):
                # OpenHands style: Explicit Thought, then Code
                block = f"step_{step_count}\n"
                if event.thought:
                    block += f"💭 THOUGHT: {event.thought}\n"
                block += f"🛠️ ACTION: {event.code}"
                context_lines.append(block)
                step_count += 1

            elif isinstance(event, BrowserObservation):
                # OpenHands style: The raw output of the tool
                status = "SUCCESS" if event.last_action_success else "FAIL"
                if event.error:
                    content = f"❌ ERROR: {event.error}"
                else:
                    # We summarize the tree in history to save tokens, but keep the URL
                    content = f"URL: {event.url}\n(Tree content hidden for brevity - see Current State)"
                
                context_lines.append(f"👀 OBSERVATION [{status}]:\n{content}")
                context_lines.append("---")

            elif isinstance(event, Observation):
                # Generic observation
                context_lines.append(f"👀 OBSERVATION: {event.content}")
                context_lines.append("---")
        
        return "\n".join(context_lines)

    # --- EXISTING HELPER METHODS (Preserved to maintain logic) ---

    def _detect_loops(self) -> Optional[str]:
        """Detect if agent is stuck in a loop and return a warning."""
        actions = self._get_all_actions()
        if len(actions) < 4: return None
        
        # Exact repetition check
        for pattern_len in [2, 3]:
            if len(actions) >= pattern_len * 2:
                recent = actions[-pattern_len:]
                previous = actions[-pattern_len*2:-pattern_len]
                if recent == previous:
                    return f"⚠️ SYSTEM ALERT: Loop detected. You performed {' -> '.join(recent)} twice. STOP and try a different strategy."
        
        # Stuck on one action
        if len(actions) >= 5:
            if actions[-1] == actions[-2] == actions[-3]:
                 return f"⚠️ SYSTEM ALERT: You are repeating `{actions[-1]}`. It is not working. Do not do this again."
        return None

    def _get_all_actions(self) -> List[str]:
        return [e.code for e in self.history if isinstance(e, BrowserAction)]
        
    def get_last_n_actions(self, n: int = 3) -> List[str]:
        actions = self._get_all_actions()
        return actions[-n:] if len(actions) >= n else actions

    def is_repeating_actions(self) -> bool:
        return self._detect_loops() is not None