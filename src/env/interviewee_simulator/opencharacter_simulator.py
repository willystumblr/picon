from litellm.cost_calculator import completion_cost
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.env.personas.human_simulacra.hs_agents import Top_agent
import logging
from dotenv import load_dotenv
import time
import os
import re
import torch
import gc
from transformers import AutoTokenizer

class OpenCharacterSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        assert 'model_path' in kwargs, "OpenCharacter requires (path-like, either huggingface repo OR local path) model parameter"
        assert 'persona' in kwargs, "OpenCharacter requires persona parameter"
        assert 'profile' in kwargs, "OpenCharacter requires profile parameter"
        assert 'simulator_model' in kwargs, "OpenCharacter requires simulator_model parameter"
        assert 'port' in kwargs, "OpenCharacter requires port parameter"
        super().__init__(**kwargs)
        self.client_or_model = kwargs['model_path']
        
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_path'])
        self.history = [{
            "role": "system",
            "content": (
                "You are an AI character with the following Persona.\n\n"
                f"# Persona\n{kwargs['persona']}\n\n"
                f"# Character Profile\n{kwargs['profile']}\n\n"
                "Please stay in your character and comply with the Persona and "
                "Character Profile while being helpful and harmless."
            ),
        }]
        self.name = re.search(r'^Name:\s*(.+)$', kwargs['profile'], flags=re.MULTILINE).group(1).strip() if self.name is None else self.name
        self.simulator_model = kwargs['simulator_model']
        self.host = kwargs.get('simulator_host', 'localhost')
        self.port = kwargs['port']
        self.max_tokens = self.get_max_token()
        
    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append({
                "role": "user",
                "content": message
            })
        self._truncate_history()

        res = get_completion(
            model=self.simulator_model,
            messages=self.history,
            reasoning_effort="low",
            api_base=f"http://{self.host}:{self.port}/v1",
            max_tokens=1024,
            temperature=0.9,
            top_p=0.9,
        )
        

        response = res.choices[0].message.content.strip()
        self.history.append({
            "role": "assistant",
            "content": response
        })
        
        self._ai_check(message, response)

        return IntervieweeResponse(question=message, content=response)
