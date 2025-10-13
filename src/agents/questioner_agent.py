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
            model=kwargs.get('model', "gemini/gemini-2.5-flash")
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

    def act(self, observation: Observation, **kwargs) -> Action:
        if observation.observation_type == "tool_output":
            assert 'tool_calls' in self.memory[-1] and self.memory[-1]['tool_calls'] is not None, "Last memory entry must be a tool call."
            for tool_output in observation.tool_output:
                idx = next((i for i, entry in enumerate(self.memory) if 'tool_calls' in entry and entry['tool_calls'] is not None and any(tc['id'] == tool_output.tool_call_id for tc in entry['tool_calls'])), None)
                if idx is not None:
                    self.update_memory(
                        role="tool",
                        tool_call_id=tool_output.tool_call_id,
                        name=tool_output.tool_name,
                        content=str(tool_output.output),
                        index=idx+1 # insert right after the tool call
                    )
        res = get_completion(
            model=self.model,
            messages=self.memory,
            reasoning_effort="low"
        )
        self._calculate_cost(res)
        question = res.choices[0].message.content.strip()
        self.update_memory(role="assistant", content=question)
        return Action(agent=self.role, action_type="respond", content=question)

