from typing import List, Dict
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

class QuestionerAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "questioner"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash"),
            port=kwargs.get('port', None)
        )

    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.memory[0]['content'] = self.memory[0]['content'].format(current_date=cutoff_date)

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

    def act(self, verdict: Dict = None) -> Action:
        if verdict:
            self.update_memory(
                role="user",
                content=f"[INSTRUCTION] The following is the evaluator's verdict on the previous QA, evaluating its consistency with the previous conversation history. "
                        "You may refer to this verdict for the next question formulation.\n\n"
                        f"Verdict: {json.dumps(verdict)}"
            )
        completion_kwargs = dict(
            model=self.model,
            messages=self.memory,
            reasoning_effort="low"
        )
        if self.model.startswith("hosted_vllm/"):
            assert self.port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"
        res = get_completion(**completion_kwargs)
        self._calculate_cost(res)
        # logging.info(f"[REASONING TRACE] {self.role} {res.choices[0].message.reasoning_content}")
        question = res.choices[0].message.content.strip()
        self.update_memory(role="assistant", content=question)
        return Action(agent=self.role, action_type="respond", content=question)

