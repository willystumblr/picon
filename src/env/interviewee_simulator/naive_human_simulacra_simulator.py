from litellm.cost_calculator import completion_cost
from streamlit import json
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.env.personas.human_simulacra.hs_agents import Naive_Agent
import logging
from dotenv import load_dotenv
import time
import os


class NaiveHumanSimulacraSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'name' in kwargs, "Human Simulacra requires name parameter"
        assert 'simulator_model' in kwargs, "Human Simulacra requires simulator_model parameter"
        self.client_or_model = Naive_Agent(character_name=kwargs['name'], model=kwargs['simulator_model']) ### has its own chat history
        
    
    def _get_response(self, message: str) -> IntervieweeResponse:
        response = self.client_or_model.send_message(message)
        
        self._ai_check(message, response)
        return IntervieweeResponse(
            question=message,
            content=response
        )
        
    
    def calculate_cost(self) -> float:
        return self.client_or_model.calculate_cost() + self.cost