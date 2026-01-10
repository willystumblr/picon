from litellm.cost_calculator import completion_cost
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.env.personas.human_simulacra.hs_agents import Top_agent
import logging
from dotenv import load_dotenv
import time
import os


class HumanSimulacraSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'name' in kwargs, "Human Simulacra requires name parameter"
        assert 'simulator_model' in kwargs, "Human Simulacra requires simulator_model parameter"
        self.client_or_model = Top_agent(character_name=kwargs['name'], model=kwargs['simulator_model']) ### has its own chat history
        
        
    def get_response(self, message: str) -> IntervieweeResponse:
        error_message = ""
        for attempt in range(3):  # Retry up to 3 times
            try:
                return self._get_response(message)
            except Exception as e:
                # if "AI Detected", immediately raise value error
                if str(e) == "AI Detected":
                    raise e
                logging.error(f"Error getting response: {e}. Attempt {attempt + 1} of 3.")
                error_message += f"Attempt {attempt + 1}: {str(e)}\n"
                time.sleep(2)  # Wait before retrying
        logging.error(f"All attempts failed. Errors:\n{error_message}")
        raise RuntimeError(f"Failed to get persona response after 3 attempts. Errors:\n{error_message}")
    
    def _get_response(self, message: str) -> IntervieweeResponse:
        response = self.client_or_model.send_message(message)
        
        self._ai_check(message, response)
        return IntervieweeResponse(question=message, content=response)
        
    
    def calculate_cost(self) -> float:
        return self.client_or_model.calculate_cost() + self.cost