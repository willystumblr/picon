from litellm.cost_calculator import completion_cost
from litellm import get_max_tokens
import tiktoken
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
import logging

class DeepPersonaSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'simulator_model' in kwargs, "DeepPersonaSimulator requires simulator_model parameter"
        assert 'persona' in kwargs, "DeepPersonaSimulator requires persona parameter"
        
        self.client_or_model = kwargs['simulator_model']
        self.persona = kwargs['persona']
        self.history = [
            {
                "role": "system",
                "content": (
                    "Given the following user profile and request, generate a "
                    "personalized response tailored to the user’s background and attributes."
                ),
            },
            {
                "role": "user",
                "content": f"User profile: {self.persona};"
            }
        ]
        self.name = kwargs.get('name', 'whatchamacallit')
        self.host = kwargs.get('simulator_host', None)
        self.port = kwargs.get('port', None)
        self.max_tokens = get_max_tokens(self.client_or_model)

    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append({
            "role": "user",
            "content": f"user request: {message}"
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
