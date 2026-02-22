
# https://huggingface.co/datasets/proj-persona/PersonaHub

from litellm.cost_calculator import completion_cost
from streamlit import json
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
import logging
from dotenv import load_dotenv
import time
import os


class PersonaHubSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'persona' in kwargs, "Persona Hub requires persona parameter"
        assert 'simulator_model' in kwargs, "Persona Hub requires simulator_model parameter"
        print(f"You are {kwargs['persona']}.\n\n")
        self.history = [{
            "role": "system",
            "content": (
                f"You are {kwargs['persona']}\n\n"
                "Please stay in your character and comply with the persona."
                "Don't mention that you are an AI model."
            ),
        }]
        self.client_or_model = kwargs['simulator_model']
        self.max_tokens = self.get_max_token()
        
    
    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append({
                "role": "user",
                "content": message
            })
        self._truncate_history()
        
        res = get_completion(
            model=self.client_or_model,
            messages=self.history,
            reasoning_effort="low",
            # temperature=0.9,
            # top_p=0.9,
        )
        cost = completion_cost(completion_response=res)
        self.cost += cost

        response = res.choices[0].message.content.strip()
        self.history.append({
            "role": "assistant",
            "content": response
        })
        
        self._ai_check(message, response)

        return IntervieweeResponse(question=message, content=response)