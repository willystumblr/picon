from typing import List, Dict, Any, Literal
import re
import time
from pydantic import BaseModel, Field, ValidationError
import logging
import litellm
import os
import json
import time
from src.agents.base_agent import Agent
from src.utils import get_completion
from src.schemas import Action, Observation
litellm.drop_params = True

class EvaluatorAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "evaluator"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash")
        )

    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.memory[0]['content'] = self.memory[0]['content'].format(cutoff_date=cutoff_date)

    def update_memory(self, **kwargs) -> None:
        if kwargs.get('index') is not None:
            index = kwargs.pop('index')
            assert 0 <= index <= len(self.memory), "Index out of range"
            # insert at specific index
            if index == len(self.memory):
                self.memory.append(kwargs)
            else:
                self.memory.insert(index, kwargs)
        else:
            self.memory.append(kwargs) # typically role and content

    def act(self) -> Action:
        class EvaluationResponse(BaseModel):
            verdict: Literal['conflict', 'plausible']  # 'conflict' or 'plausible'
            rationale: str  # explanation for the verdict
            ground: Literal['internal', 'external'] = Field(description="Ground for the verdict. If `internal`, the verdict is based on the internal context (i.e., the conversation history without external information). If `external`, it is based on external web search results.")
        
        res = get_completion(
            model=self.model,
            messages=self.memory,
            reasoning_effort="low",
            response_format=EvaluationResponse
        )
        self._calculate_cost(res)
        # logging.info(f"[REASONING TRACE] {self.role} {res.choices[0].message.reasoning_content}")
        res_eval = EvaluationResponse.model_validate_json(res.choices[0].message.content)  # validate response format

        return Action(agent=self.role, action_type="respond", content=res_eval.model_dump())

