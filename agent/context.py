from dataclasses import dataclass, field
from typing import Any

from agent.states import AgentState


@dataclass
class AgentContext:

    goal: str

    state: AgentState = AgentState.DECIDE

    step: int = 0

    previous_response_id: str | None = None

    pending_call: Any | None = None

    observation: str | None = None

    final_answer: str | None = None

    history: list = field(default_factory=list)