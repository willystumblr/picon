from litellm.cost_calculator import completion_cost
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.env.personas.human_simulacra.hs_agents import Top_agent
import logging
from dotenv import load_dotenv

class Twin2K500Simulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'simulator_model' in kwargs, "Twin2K500 requires simulator_model parameter"
        assert 'persona' in kwargs, "Twin2K500 requires persona parameter"
        
        self.client_or_model = kwargs['simulator_model']
        
        self.history = [{
            "role": "system",
            "content": (
                "You are an AI assistant. Your task is to answer the ’New Survey Question’ as "
                "if you are the individual described in the ’Persona Profile’ (which contains their "
                "past survey responses). Remain consistent with the persona’s previous answers"
                "and stated characteristics. Carefully follow any instructions provided for the new "
                f"question, including formatting requirements.\n\n{kwargs['persona']}"
            ),
        }]
        self.name = kwargs.get('name', 'whatchamacallit')
        self.simulator_model = kwargs['simulator_model']
        self.host = kwargs.get('simulator_host', None)
        self.port = kwargs.get('port', None)
        self.max_tokens = self.get_max_token()
        
    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append({
                "role": "user",
                "content": message
            })
        self._truncate_history()    
        completion_kwargs = {
            "model": self.simulator_model,
            "messages": self.history,
            "reasoning_effort": "low",
        }
        if self.host and self.port:
            completion_kwargs['api_base'] = f"http://{self.host}:{self.port}/v1"
        res = get_completion(**completion_kwargs)
        

        response = res.choices[0].message.content.strip()
        self.history.append({
            "role": "assistant",
            "content": response
        })
        
        self._ai_check(message, response)

        return IntervieweeResponse(question=message, content=response)
