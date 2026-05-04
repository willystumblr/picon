from pydantic import BaseModel
from typing import Literal, Dict, Any, List

class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    details: List | Dict | Any | None = None

class Action(BaseModel):
    agent: str | None = None
    action_type: Literal["respond", "next_agent", "tool_call", "finish"]
    content: Any | None = None
    target_agent: str | None = None
    tool_call: ToolCall | None = None

class IntervieweeResponse(BaseModel):
    question: str
    content: str

class ToolOutput(BaseModel):
    tool_call_id: str
    tool_name: str
    arguments: Dict[str, Any] | None = None  # Optional for backward compatibility with saved data
    output: str |List | List[Dict] | Any

class Observation(BaseModel):
    observation_type: Literal["interviewee_response", "tool_output"]
    response: IntervieweeResponse | None = None
    tool_output: List[ToolOutput] | None = None # if we did the tool call

class Turn(BaseModel):
    type: Literal['get_to_know', 'main_interrogation', 'repeat']
    agent_action: List[Action] | None = None
    environment_observation: List[Observation] = []

class State(BaseModel):
    current_turn: int
    current_observation: Observation | None = None
    history: List[Turn] = []