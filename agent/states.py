from enum import Enum, auto


class AgentState(Enum):

    DECIDE = auto()
    ACT = auto()
    OBSERVE = auto()

    DONE = auto()
    ERROR = auto()