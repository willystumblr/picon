from typing import Dict, List, Literal
from pydantic import BaseModel
from src.agents.base_agent import Agent
from src.schemas import Action, ToolCall
from src.utils import get_completion
import logging
import json
import time

class WebSearchAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(
            role=kwargs.get('role', "web_search"),
            system_message=kwargs.get('system_message', ""),
            model=kwargs.get('model', "gemini/gemini-2.5-flash"),
            port=kwargs.get('port', None)
        )
        self.tools = kwargs.get('tools', [])
        self.cutoff_date = time.strftime("%B %d, %Y") # default to current date
    
    def set_cutoff_date(self, cutoff_date: str) -> None:
        self.cutoff_date = cutoff_date
    
    def act(self, message: Dict, history: List) -> Action:
        if not self.tools:
            return Action(
                action_type="next_agent",
                target_agent="questioner"
            )
        # for entity_claim in message:
        entity = message.get('entity', 'no entity')
        claims = message.get('claims', ['no claim'])
        rationale = message.get('rationale', '')
        if entity and entity != 'no entity':
            prompt_format = f"Does the following entity-claims pair need to be verified with web-search?\nEntity: {entity}\nClaims: {claims}\n\nRationale: {rationale}\nCutoff Date: {self.cutoff_date}"
        elif entity == 'no entity':
            prompt_format = f"Does the following claim need to be verified with web-search?\nClaim: {claims}\nCutoff Date: {self.cutoff_date}"
        while True:
            class ResponseFormat(BaseModel):
                type: Literal["yes", "no"]
            prompt = f"Conversation History:\n\n"
            for qa in history:
                prompt += f"Interviewer: \"{qa['question']}\"\nInterviewee: \"{qa['answer']}\"\n\n"
            prompt += prompt_format
            res = get_completion(
                model=self.model,
                messages=self.memory + [{"role": "user", "content": prompt}],
                tool_choice="none",
                tools=self.tools,
                reasoning_effort="low",
                response_format=ResponseFormat,
            )
            self._calculate_cost(res)
            proceed_to_web_search = ResponseFormat.model_validate_json(res.choices[0].message.content).type.lower()
            if proceed_to_web_search in ['yes', 'no']:
                break

        if proceed_to_web_search == 'yes': # , and the claim: {claim}, with rationale: {rationale}
            prompt = f"Search {entity} with a proper tool." if entity and entity!="no entity" else f"Search the claim: {claims} with a proper tool."
            completion_kwargs = dict(
                model=self.model,
                messages=self.memory + [{"role": "user", "content": prompt}], # no memory needed
                tool_choice="required",
                tools=self.tools,
                reasoning_effort="low",
                parallel_tool_calls=False,
            )
            if self.model.startswith("hosted_vllm/"):
                assert self.port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.port}/v1"
            
            res = get_completion(**completion_kwargs)
            self._calculate_cost(res)
            res_ = res.choices[0].message.model_dump()
            if 'tool_calls' in res_ and res_['tool_calls']:
                tool_call = res_['tool_calls'][0]
                tool_name = tool_call['function']['name']
                arguments = json.loads(tool_call['function']['arguments'])
                return Action(
                    agent=self.role,
                    action_type="tool_call",
                    tool_call=ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        details=res_
                    )
                )
        else:
            logging.warning(f"WebSearchAgent decided no web search needed for entity: {entity}, claims: {claims}.")
            return None