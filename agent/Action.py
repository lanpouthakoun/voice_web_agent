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
    """Renders event history into prompts for the LLM with loop detection."""
    
    def __init__(self, history: List[Event]):
        self.history = history

    def get_prompt_context(self, max_events: int = 10) -> str:
        """
        Renders history with:
        1. Loop detection warnings
        2. Action frequency summary
        3. Clear action → result pairing
        """
        if not self.history:
            return "No previous actions."
        
        lines = []
        
        # 1. Check for loops and add warning at the top
        loop_warning = self._detect_loops()
        if loop_warning:
            lines.append(loop_warning)
            lines.append("")
        
        # 2. Add action frequency summary if there are many actions
        all_actions = self._get_all_actions()
        if len(all_actions) > 5:
            freq_summary = self._get_action_frequency_summary(all_actions)
            if freq_summary:
                lines.append(freq_summary)
                lines.append("")
        
        # 3. Show recent action → result pairs
        lines.append("## Recent Actions:")
        recent_context = self._format_recent_events(max_events)
        lines.append(recent_context)
        
        return "\n".join(lines)

    def _detect_loops(self) -> Optional[str]:
        """Detect if agent is stuck in a loop and return a warning."""
        actions = self._get_all_actions()
        
        if len(actions) < 4:
            return None
        
        # Check for exact repetition of last 2-4 actions
        for pattern_len in [2, 3, 4]:
            if len(actions) >= pattern_len * 2:
                recent = actions[-pattern_len:]
                previous = actions[-pattern_len*2:-pattern_len]
                if recent == previous:
                    return f"""⚠️ LOOP DETECTED: You are repeating the same {pattern_len} actions!
Pattern: {' → '.join(recent)}
This pattern has occurred at least twice. TRY SOMETHING DIFFERENT.
- Look at the accessibility tree for NEW elements
- Maybe a modal/popup is blocking - look for close/dismiss buttons
- Maybe the page layout changed - scroll or look for different elements"""
        
        # Check for high repetition of single action
        if len(actions) >= 6:
            last_6 = actions[-6:]
            counts = Counter(last_6)
            most_common, count = counts.most_common(1)[0]
            if count >= 4:
                return f"""⚠️ STUCK: You've done `{most_common}` {count} times in the last 6 actions!
This action is not achieving your goal. Try a DIFFERENT approach:
- Is there a different element ID you should target?
- Is something blocking the element (modal, overlay)?
- Do you need to scroll to find the right element?"""
        
        return None

    def _get_action_frequency_summary(self, actions: List[str]) -> str:
        """Summarize how many times each action type was used."""
        # Simplify actions to their type for counting
        simplified = []
        for action in actions:
            if action.startswith("click("):
                # Extract the ID
                simplified.append(action)
            elif action.startswith("fill("):
                simplified.append("fill(...)")
            elif action.startswith("scroll("):
                simplified.append("scroll(...)")
            elif action.startswith("noop"):
                simplified.append("noop/wait")
            else:
                simplified.append(action.split("(")[0] + "(...)")
        
        counts = Counter(simplified)
        
        # Only show if there's repetition
        repeated = [(action, count) for action, count in counts.most_common(5) if count > 1]
        if not repeated:
            return ""
        
        lines = ["## Action Summary (you may be repeating yourself):"]
        for action, count in repeated:
            lines.append(f"  - `{action}`: {count} times")
        
        return "\n".join(lines)

    def _get_all_actions(self) -> List[str]:
        """Get all action codes from history."""
        return [e.code for e in self.history if isinstance(e, BrowserAction)]

    def _format_recent_events(self, max_events: int) -> str:
        """Format recent events as action → result pairs."""
        context_lines = []
        recent_events = self.history[-max_events:]
        
        i = 0
        step_num = max(1, len(self._get_all_actions()) - max_events // 2)  # Approximate step number
        
        while i < len(recent_events):
            event = recent_events[i]
            
            if isinstance(event, BrowserAction):
                action_str = f"Step {step_num}: `{event.code}`"
                
                # Look for the next observation (result)
                result_str = "(no result recorded)"
                if i + 1 < len(recent_events):
                    next_event = recent_events[i + 1]
                    if isinstance(next_event, BrowserObservation):
                        result_str = self._format_observation_result(next_event, event.code)
                        i += 1
                
                context_lines.append(f"{action_str}")
                context_lines.append(f"   → {result_str}")
                step_num += 1
                
            elif isinstance(event, BrowserObservation):
                # Standalone observation (initial page load)
                if i == 0:  # Only show if it's the first event
                    context_lines.append(f"Initial page: {event.url}")
                    
            elif isinstance(event, MessageAction):
                context_lines.append(f"💬 Sent to user: {event.content}")
            
            i += 1
        
        return "\n".join(context_lines)

    def _format_observation_result(self, obs: BrowserObservation, action_code: str) -> str:
        """Format an observation as a result of an action."""
        
        if obs.error:
            return f"❌ FAILED: {obs.error}"
        
        if not obs.last_action_success:
            hint = self._get_hint_for_no_change(action_code)
            return f"⚠️ NO EFFECT{hint}"
        
        # Success case
        if "fill(" in action_code:
            return f"✓ Text entered (NOT submitted yet - need click/Enter)"
        elif "click(" in action_code:
            return f"✓ Clicked (URL: {self._shorten_url(obs.url)})"
        elif "goto(" in action_code:
            return f"✓ Navigated to {self._shorten_url(obs.url)}"
        elif "press(" in action_code:
            return f"✓ Key pressed"
        elif "scroll(" in action_code:
            return f"✓ Scrolled"
        elif "noop" in action_code:
            return f"✓ Waited"
        else:
            return f"✓ Done"

    def _shorten_url(self, url: str, max_len: int = 60) -> str:
        """Shorten URL for display."""
        if len(url) <= max_len:
            return url
        return url[:max_len-3] + "..."

    def _get_hint_for_no_change(self, action_code: str) -> str:
        """Provide helpful hints when an action had no effect."""
        if "fill(" in action_code:
            return " - element may not exist or already has this text"
        elif "click(" in action_code:
            return " - element may not be clickable, check the ID"
        elif "scroll(" in action_code:
            return " - already at scroll boundary"
        return ""

    def get_last_n_actions(self, n: int = 3) -> List[str]:
        """Get the last N action codes."""
        actions = self._get_all_actions()
        return actions[-n:] if len(actions) >= n else actions

    def is_repeating_actions(self) -> bool:
        """Detect if the agent is stuck repeating actions."""
        actions = self._get_all_actions()
        
        if len(actions) < 4:
            return False
        
        # Check for 2-action loop
        if actions[-2:] == actions[-4:-2]:
            return True
        
        # Check for same action 3+ times in last 5
        if len(actions) >= 5:
            last_5 = actions[-5:]
            counts = Counter(last_5)
            if counts.most_common(1)[0][1] >= 3:
                return True
        
        return False

    def get_loop_info(self) -> Optional[Tuple[List[str], int]]:
        """Returns (pattern, repetition_count) if a loop is detected."""
        actions = self._get_all_actions()
        
        for pattern_len in [2, 3, 4]:
            if len(actions) >= pattern_len * 2:
                pattern = actions[-pattern_len:]
                # Count how many times this pattern appears at the end
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