from litellm.cost_calculator import completion_cost
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
import logging
from dotenv import load_dotenv
import time
import os


class BaseIntervieweeSimulator:
    def __init__(self, **kwargs):
        load_dotenv()
        self.type = kwargs.get('baseline_name')
        self.name = kwargs.get('name', None)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.__nhd_prompt = open(f"{project_root}/src/agents/prompts/nhd_detector.txt", "r").read()
        self.__nhd_model = kwargs.get('nhd_model', "gemini/gemini-2.5-flash")
        self.__nhd_port = kwargs.get('nhd_port', None)
        if self.__nhd_model.startswith("hosted_vllm/"):
            assert self.__nhd_port is not None, "NHD port must be provided for hosted_vllm models"
        self.cost = 0.0
        
        
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
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def calculate_cost(self) -> float:
        return self.cost