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
from transformers import AutoTokenizer
import torch
import gc

class ConsistentLLMSimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'model_path' in kwargs, "Consistent LLM requires (path-like, either huggingface repo OR local path) model parameter"
        assert 'persona' in kwargs, "Consistent LLM requires persona parameter"
        assert 'name' in kwargs, "Consistent LLM requires name parameter"
        assert 'instruction' in kwargs, "Consistent LLM requires instruction parameter"
        assert 'counterpart_name' in kwargs, "Consistent LLM requires counterpart_name parameter"
        assert 'simulator_model' in kwargs, "Consistent LLM requires simulator_model parameter"
        assert 'port' in kwargs, "Consistent LLM requires port parameter"
        
        self.client_or_model = kwargs['model_path']
        self.simulator_model = kwargs['simulator_model']
        self.host = kwargs.get('simulator_host', 'localhost')
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_path'])
        self.persona = kwargs['persona']
        self.history = []
        self.instruction = kwargs['instruction']
        self.counterpart_name = kwargs['counterpart_name']
        self.prompt_flag = "Your conversation so far is below:\nConversation: \n"
        self.port = kwargs['port']
        
    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append(f"Interviewer: {message}")
            
        while True:
            user_content = self.prompt_flag + '\n'.join(self.history) + self.instruction
            messages = [
                {"role": "system", "content": self.persona},
                {"role": "user", "content": user_content},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )

            if input_ids.shape[1] + 1024 <= 8192:  # assuming model max position is 8192
                break

            self.history = self.history[2:]  # drop the oldest message

        res = get_completion(
            model=self.simulator_model,
            messages=messages,
            reasoning_effort="low",
            api_base=f"http://{self.host}:{self.port}/v1",
            max_tokens=1024
        )
        # response = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)    
        response = res.choices[0].message.content.strip()
        self.history.append(f"{self.name}: {response}")
        
        self._ai_check(message, response)
        return IntervieweeResponse(question=message, content=response)