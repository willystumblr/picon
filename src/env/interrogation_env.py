import time
from typing import List, Dict, Any, Literal
from src.agents.base_agent import Agent
from src.env.interviewee_simulator.simulator_factory import get_interviewee_simulator
from src.agents.agent_factory import get_agent
from src.schemas import State, Action, Observation, Turn, ToolOutput, IntervieweeResponse
from src.tools.address_locator import GoogleGeocodeValidate
from src.tools.web_search import GoogleClaimSearch
from pydantic import BaseModel, Field
from src.utils import read_json, write_json, get_completion
import logging
from litellm.cost_calculator import completion_cost
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import random
import asyncio
import litellm
#litellm._turn_on_debug()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

REPEAT_PROMPT = """You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""


class InterrogationEnv:
    def __init__(
        self, 
        agents: Dict[str, Agent] = {},
        baseline_name: str = "characterai",
        tools: List[Dict[str, Any]] = [],
        max_turns: int = 20,
        question_path: str = "src/env/wvs_orthogonal_questions.json",
        instruction_path: str = "src/env/interrogation_instruct.txt",
        **kwargs
        ):
        
        # random seed for reproducibility
        seed = kwargs.get('question_seed', 42)
        local_rng = random.Random(seed)
        
        self.tools = tools
        if not agents:
            logging.warning("No agents provided. Initializing default agents.")
            agents = {
                "questioner": get_agent("questioner", f"{project_root}/src/agents/prompts/questioner.txt", model='gemini/gemini-2.5-flash', port=None),
                "extractor": get_agent("claim_extractor", f"{project_root}/src/agents/prompts/claim_extractor_prompt.txt") if kwargs.get('use_claim_extractor', True) else get_agent("entity_extractor", f"{project_root}/src/agents/prompts/entity_extractor.txt", model='gemini/gemini-2.5-flash', port=None),
                "web_search": get_agent("web_search", f"{project_root}/src/agents/prompts/websearch_prompt.txt", model='gemini/gemini-2.5-flash', port=None),
                "evaluator": get_agent("evaluator", f"{project_root}/src/agents/prompts/evaluator_prompt.txt", model='gemini/gemini-2.5-flash', port=None),
            }
        self.agents = agents
        if 'web_search' in self.agents and not self.tools:
            logging.warning("No tools provided for web search agent.")
        if 'web_search' in self.agents and not self.agents['web_search'].tools:
            logging.info("Setting web search agent tools from environment.")
            self.agents['web_search'].tools = [tool.get_info() for tool in self.tools.values()]
        
        self.interviewee = get_interviewee_simulator(
            baseline_name=baseline_name,
            **kwargs # simulator specific args (character_id, user_id, name for characterai; model_path, persona, profile for opencharacter; name for human_simulacra)
        )
        self.max_turns = max_turns
        questions = read_json(question_path)
        local_rng.shuffle(questions)
        self.predefined_questions = questions
        self.instruction = open(instruction_path).read()
        self.state = State(current_turn=0, history=[])
        self.cutoff_date = time.strftime("%B %d, %Y")
        self.start_time = None
        self.env_cost = 0.0

        self.agents['questioner'].set_cutoff_date(self.cutoff_date)
        self.agents['web_search'].set_cutoff_date(self.cutoff_date)
        self.agents['evaluator'].set_cutoff_date(self.cutoff_date)
        
        # repeat
        self.repeat_score = 0
        self.repeat_results = []

        # confimation (for affirmative)
        with open(f"{project_root}/src/agents/prompts/confirmation_prompt.txt") as f:
            self.confirmation_prompt = f.read()
        self.confirmation_qa_lists = []
        
        # evasiveness
        self.evasive_prompt = open(f"{current_dir}/abstain_analysis_prompt.txt", "r").read()
        self.evasive_score = 0
        self.evasive_results = []
        
        # affirmative
        self.affirmative_score = 0
        self.affirmative_count = 0
        self.refuted_count = 0
        self.affirmative_results = []
        
        # consistency
        self.internal_count = 0
        self.external_count = 0
        self.internal_conflict = 0
        self.external_conflict = 0
        self.internal_plausible = 0
        self.external_plausible = 0
        self.internal_conflict_verdicts = []
        self.external_conflict_verdicts = []
        self.first_conflict_turn = None

    def invoke_tool(self, action: Action) -> Observation | None:
        if action.action_type == "tool_call":
            tool_name = action.tool_call.tool_name
            if tool_name not in self.tools:
                logging.error(f"Tool {tool_name} not found.")
                return self.state, True
            
            
            tool = self.tools[tool_name]
            tool_output = tool.invoke(**action.tool_call.arguments) # if 'claim' in action.tool_call.arguments else tool.invoke_batch(**action.tool_call.arguments) # batch for claims list
            logging.info(f"[TOOL OUTPUT] {tool_name}: {tool_output[:100]}...") # print first 100 chars
            
            output = ToolOutput(
                tool_call_id=action.tool_call.details.get('tool_calls')[0].get('id'),
                tool_name=tool_name,
                output=tool_output
            )
            return output
        return None

    def reset(self):
        """run predefined questions to initialize the interview state"""
        self.start_time = time.time()
        self.instruction = self.instruction.format(cutoff_date=self.cutoff_date)
        # feed interview instruction to the interviewee
        logging.info(f"[INSTRUCTION] {self.instruction}")
        response = self.interviewee.get_response(self.instruction)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
        # run predefined questions
        for i, q in enumerate(self.predefined_questions):
            logging.info(f"[QUESTION] {q['question']}")
            response = self.interviewee.get_response(q['question'])
            logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")    
            # update state
            action = Action(action_type="respond", content=q['question'])
            res_observation = Observation(
                observation_type="interviewee_response",
                response=response
            )
            
            
            self.state.current_observation = res_observation
            self.state.current_turn += 1
            
            self.agents['evaluator'].update_memory(role="assistant", content=q['question'])
            self.agents['questioner'].update_memory(role="assistant", content=q['question'])
            self.agents['evaluator'].update_memory(role="user", content=response.content)
            self.agents['questioner'].update_memory(role="user", content=response.content)
            
            entity_action, web_observations, web_search_actions = self.check_external(
                f"Question: {q['question']}\nResponse: {response.content}"
            )
            # for verdict_action in verdict_actions:
            #     self.agents['questioner'].update_memory(**{"role":"assistant", "content": str(verdict_action.content)}) ####### 여기 #######
            if web_search_actions:
                turn = Turn(type='get_to_know', agent_action=[entity_action, *web_search_actions], environment_observation=[res_observation, *web_observations])
            else:
                turn = Turn(type='get_to_know', agent_action=[entity_action], environment_observation=[res_observation])
            
            self.state.history.append(turn)
        return self.state

    def check_external(self, message : str) -> tuple[Action, List[Observation], List[Action]]:
        # 1. Extractor first extracts the entity & claim to verify
        observations = []
        next_action = self.agents["extractor"].act(message)
        logging.info(f"[ACTION] Extractor: {next_action.action_type} - {next_action.content if next_action.content else next_action.target_agent}")
        if next_action.action_type == "next_agent":
            if next_action.target_agent != "questioner":
                logging.error("Extractor can only pass to Questioner.")
                return self.state, True
            logging.info("No entity or claim extracted. Passing to Questioner.")
            filtered_actions = []
        elif next_action.action_type == "respond": # should be respond with entity & claim
            if next_action.content is None:
                logging.error("Extractor must respond with entity & claim.")
                return self.state, True

            list_of_extractions = [ext.model_dump() for ext in next_action.content] # list of ExtractorResponse
            history = []
            for turn in self.state.history:
                # last element is interviewee response
                qa_pair = turn.environment_observation[0].response
                history.append({
                    "question": qa_pair.question,
                    "answer": qa_pair.content
                })
            # 2. Web Search (optional)
            with ThreadPoolExecutor(max_workers=len(list_of_extractions)) as executor:
                web_search_actions = list(executor.map(self.agents['web_search'].act, list_of_extractions, [history]*len(list_of_extractions)))
            filtered_actions = []
            filtered_actions_indices = []
            for i, action in enumerate(web_search_actions): 
                if action and action.action_type == "tool_call":
                    filtered_actions.append(action)
                    filtered_actions_indices.append(i)

            if filtered_actions:
                with ThreadPoolExecutor(max_workers=len(filtered_actions)) as executor:
                    tool_outputs = list(executor.map(self.invoke_tool, filtered_actions))

                observations.append(Observation(
                    observation_type="tool_output",
                    tool_output=tool_outputs
                ))
                
                for i, output in enumerate(tool_outputs):
                    sub_message = [
                        {
                            "role": "user",
                            "content": message
                        },
                        filtered_actions[i].tool_call.details,
                        {
                            "role": "tool",
                            "tool_call_id": output.tool_call_id,
                            "name": output.tool_name,
                            "content": f"Entity:{list_of_extractions[filtered_actions_indices[i]]['entity']}\nClaims: {list_of_extractions[filtered_actions_indices[i]]['claims']}\nRationale: {list_of_extractions[filtered_actions_indices[i]]['rationale']}\nSearch Result:{str(output.output)}"
                            #"content": f"Entity:{list_of_extractions[filtered_actions_indices[i]]['entity']}\nSearch Result:{str(output.output)}"
                        }
                    ]
                    """tool call 결과를 evaluator 메모리에 추가"""
                    self.agents['evaluator'].update_memory(**sub_message[1]) ####### 여기 #######
                    #tool_output = sub_message[2]
                    #tool_output['claim'] = str(list_of_extractions[filtered_actions_indices[i]]['claims'])
                    self.agents['evaluator'].update_memory(**sub_message[2]) ####### 여기 ####y
                    ###
                    messages = [
                        {
                            "role": "system",
                            "content": self.confirmation_prompt 
                        },
                    ]
                    messages.extend(sub_message)
                    completion_kwargs = dict(
                        model=self.agents['questioner'].model,
                        messages=messages,
                        reasoning_effort="low",
                    )
                    if self.agents['questioner'].model.startswith("hosted_vllm/"):
                        assert self.agents['questioner'].port is not None, "Port must be specified for hosted_vllm models."    
                        completion_kwargs['api_base'] = f"http://localhost:{self.agents['questioner'].port}/v1"
                    if self.agents['questioner'].model.startswith("claude-"):
                        completion_kwargs.pop('reasoning_effort') # claude does not support reasoning_effort
                        completion_kwargs['tools']=[] # dummy tools to avoid tool usage
                    while True:
                        res = get_completion(**completion_kwargs)
                        if res and res.choices and res.choices[0].message and res.choices[0].message.content:
                            break
                    self.env_cost += completion_cost(res) if not self.agents['questioner'].model.startswith("hosted_vllm/") else 0.0
                    confirmation_question = res.choices[0].message.content.strip()
                    if "SKIP" not in confirmation_question:
                        logging.info(f"[CONFIRMATION QUESTION] {confirmation_question}")
                        response = self.interviewee.get_response(confirmation_question)
                        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
                        observations.append(Observation(
                            observation_type="interviewee_response",
                            response=response
                        ))
                        self.agents['evaluator'].update_memory(role="assistant", content=confirmation_question) ####### 여기 #######
                        self.agents['evaluator'].update_memory(role="user", content=response.content) ####### 여기 #######
                        self.agents['questioner'].update_memory(role="assistant", content=confirmation_question) ####### 여기 #######
                        self.agents['questioner'].update_memory(role="user", content=response.content) ####### 여기 #######

                        self.confirmation_qa_lists.append({
                            "question": confirmation_question,
                            "response": response.content
                        })
                    
                    else:
                        logging.info("Confirmation question skipped as per web search agent's decision.")
            else:
                filtered_actions = []
        return next_action, observations, filtered_actions


    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation"""
        if self.state.current_turn >= self.max_turns:
            logging.warning("Max turns reached. Please reset the environment.")
            return self.state, True
        # verdict = self.state.history[-1].agent_action[-1].content if self.state.history else None
        # question_act = self.agents['questioner'].act(verdict=verdict)
        question_act = self.agents['questioner'].act()
        logging.info(f"[ACTION] Questioner: {question_act.action_type} - {question_act.content if question_act.content else question_act.tool_call.tool_name}")
        self.agents['evaluator'].update_memory(role="assistant", content=question_act.content)
        interviewee_res = self.interviewee.get_response(question_act.content)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {interviewee_res.content}")
        
        self.agents['questioner'].update_memory(role="user", content=interviewee_res.content) 
        self.agents['evaluator'].update_memory(role="user", content=interviewee_res.content)

        entity_action, observations, web_search_actions = self.check_external(f"Question: {interviewee_res.question}\nResponse: {interviewee_res.content}")

        self.state.current_turn += 1

        actions = [question_act, entity_action, *web_search_actions] if web_search_actions else [question_act, entity_action]
        res_ob = Observation(observation_type="interviewee_response", response=interviewee_res)
        observations = [res_ob, *observations]
        turn = Turn(
            type='main_interrogation',
            agent_action=actions,
            environment_observation=observations
        )
        
        self.state.history.append(turn)

        return self.state, False

    def finalize(self):
        """repeat stage: repeat the pre-defined questions to check for consistency"""
        for i, q in enumerate(self.predefined_questions):
            logging.info(f"[REPEAT QUESTION] Just to clarify, {q['question']}")
            response = self.interviewee.get_response(f"Just to clarify, {q['question']}")
            logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
            # update state
            action = Action(action_type="respond", content=q['question'])
            observation = Observation(
                observation_type="interviewee_response",
                response=response
            )
            turn = Turn(type='repeat', agent_action=[action], environment_observation=[observation])
            self.state.history.append(turn)
        if self.interviewee.type == "characterai":
            asyncio.run(self.interviewee.close())
        return self.state
    
    def evaluate(self):
        """evaluate the entire interrogation session"""
        """use threading to parallelize the four evaluation tasks at once"""
        eval_methods = [
            self.consistency_eval,
            self.repeat_eval,
            self.affirmative_eval,
            self.evasiveness_eval
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(method) for method in eval_methods]
            # Wait for all to complete
            for future in futures:
                future.result()
    
    def consistency_eval(self):
        messages_lists = []
        user_indices = []
        for i in range(len(self.agents['evaluator'].memory)):
            if self.agents['evaluator'].memory[i]['role']=='user':
                messages_lists.append(self.agents['evaluator'].memory[:i+1])
                user_indices.append(i)
        
        with ThreadPoolExecutor(max_workers=len(messages_lists)) as executor:
            results = list(executor.map(self.generate_verdict, messages_lists))
        
        for idx, (verdict, response_format) in zip(user_indices, results):
            self.env_cost += completion_cost(verdict) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
            parsed_verdict = response_format.model_validate_json(verdict.choices[0].message.content)
            turn_idx, question, user_response = self._find_turn_idx(idx)
            if parsed_verdict.ground == 'internal':
                self.internal_count += 1
                if parsed_verdict.verdict == 'conflict':
                    self.internal_conflict += 1
                    self.internal_conflict_verdicts.append({
                        "turn": turn_idx,
                        "question": question,
                        "response": user_response,
                        "rationale": parsed_verdict.rationale
                    })
                    if self.first_conflict_turn is None:
                        self.first_conflict_turn = turn_idx
                else:
                    self.internal_plausible += 1
            else:
                self.external_count += 1
                if parsed_verdict.verdict == 'conflict':
                    self.external_conflict += 1
                    self.external_conflict_verdicts.append({
                        "turn": turn_idx,
                        "question": question,
                        "response": user_response,
                        "rationale": parsed_verdict.rationale
                    })
                    if self.first_conflict_turn is None:
                        self.first_conflict_turn = turn_idx
                else:
                    self.external_plausible += 1
            
    
    def generate_verdict(self, messages: List[Dict[str, Any]]):
        class EvaluationResponseInternal(BaseModel):
            verdict: Literal['conflict', 'plausible']  # 'conflict' or 'plausible'
            rationale: str  # explanation for the verdict
            ground: Literal['internal'] = Field(description="Ground for the verdict. The verdict is based on the internal context (i.e., the conversation history without external information)")

        class EvaluationResponseExternal(BaseModel):
            verdict: Literal['conflict', 'plausible']  # 'conflict' or 'plausible'
            rationale: str  # explanation for the verdict
            ground: Literal['external'] = Field(description="Ground for the verdict. The verdict is based on external web search results.")
        
        if messages[-3]['role']=='tool':
            response_format = EvaluationResponseExternal
            ground = "external"
        else:
            response_format = EvaluationResponseInternal
            ground = 'internal'
        completion_kwargs = dict(
            model=self.agents['evaluator'].model,
            #messages=messages[:-1] + [{'role' : messages[-1]['role'] , 'content': messages[-1]['content'] + f"[conflict type] ground: {ground}"}], 
            messages=messages,
            temperature=0.7,
            response_format=response_format
        )
        if self.agents['evaluator'].model.startswith("hosted_vllm/"):
            assert self.agents['evaluator'].port is not None, "Port must be specified for hosted_vllm models."    
            completion_kwargs['api_base'] = f"http://localhost:{self.agents['evaluator'].port}/v1"    
        verdict = get_completion(**completion_kwargs)
        self.env_cost += completion_cost(verdict) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
        return verdict, response_format
    
    def _find_turn_idx(self, idx) -> int:
        """find the turn index in self.history corresponding to the idx-th user response in self.evaluator_history"""
        assert self.agents['evaluator'].memory[idx]['role'] == 'user', "The provided index does not correspond to a user message."
        user_response = self.agents['evaluator'].memory[idx]['content']
        for i, turn in enumerate(self.state.history):
            if turn.type != 'repeat':
                for env_obs in turn.environment_observation:
                    if env_obs.observation_type == "interviewee_response":
                        if env_obs.response.content == user_response:
                            question = env_obs.response.question
                            return i, question, user_response
    
    def repeat_eval(self):
        completion_kwargs_list = []
        get_to_knows = [turn for turn in self.state.history if turn.type == 'get_to_know']
        repeats = [turn for turn in self.state.history if turn.type == 'repeat']
        for original, repeat in zip(get_to_knows, repeats):
            assert original.environment_observation[0].response.question in repeat.environment_observation[0].response.question, "Mismatch in questions between original and repeat."
            
            completion_kwargs = dict(
                model=self.agents['evaluator'].model,
                messages=[
                    {"role": "system", "content": REPEAT_PROMPT},
                    {"role": "user", "content": f"Question: {original.environment_observation[0].response.question}\n\nResponse 1: {original.environment_observation[0].response.content}\nResponse 2: {repeat.environment_observation[0].response.content}"}
                ],
                reasoning_effort="low",
            )
            if self.agents['evaluator'].model.startswith("hosted_vllm/"):
                assert self.agents['evaluator'].port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.agents['evaluator'].port}/v1"
            completion_kwargs_list.append(completion_kwargs)
        
        with ThreadPoolExecutor(max_workers=len(completion_kwargs_list)) as executor:
            results = list(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list))
        for i, res in enumerate(results):
            self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
            judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while judge not in ["TRUE", "FALSE"]:
                logging.warning(f"Unexpected response for repeat score: {judge}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
                judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            self.repeat_results.append({
                "question": f"Just to clarify, {get_to_knows[i].environment_observation[0].response.question}",
                "original_response": get_to_knows[i].environment_observation[0].response.content,
                "repeated_response": repeats[i].environment_observation[0].response.content,
                "is_repeat": judge
            })
            self.repeat_score += (judge=='TRUE')
        self.repeat_score = round(self.repeat_score / len(get_to_knows), 4)
    
    def affirmative_eval(self):
        class OutputSchema(BaseModel):
            is_affirmative: Literal['true', 'false'] = Field(..., description="Whether the provided response is affirmative.")
            reason : str = Field(..., description="A brief explanation for the decision.")
        
        completion_kwargs_list = []
        for qa in self.confirmation_qa_lists: 
            completion_kwargs = dict(
                model=self.agents['evaluator'].model,
                messages=[
                    {
                        "role": "system",
                        "content": "You will be given a question and an answer. Determine whether the answer is affirmative (i.e., agrees with or confirms the question). Respond with 'true' or 'false' and provide a brief reason. If the answer is not clearly affirmative or negative (e.g., ambiguous or unclear), respond with 'false'."
                    },
                    {
                        "role": "user",
                        "content": f"Question: {qa['question']}\nAnswer: {qa['response']}"
                    }
                ],
                response_format=OutputSchema,
            )
            if self.agents['evaluator'].model.startswith("hosted_vllm/"):
                assert self.agents['evaluator'].port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.agents['evaluator'].port}/v1"
            completion_kwargs_list.append(completion_kwargs)

        with ThreadPoolExecutor(max_workers=len(completion_kwargs_list)) as executor:
            results = list(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list))
        for i, res in enumerate(results):
            self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
            output = res.choices[0].message.content if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while output is None:
                logging.warning(f"Unexpected response for affirmative eval: {output}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
                output = res.choices[0].message.content if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            try:
                parsed_output = OutputSchema.model_validate_json(output)
            except Exception as e:
                logging.error(f"Error parsing output for affirmative eval: {e}")
                continue
            self.affirmative_results.append({
                "question": self.confirmation_qa_lists[i]['question'],
                "response": self.confirmation_qa_lists[i]['response'],
                "is_affirmative": parsed_output.is_affirmative,
                "reason": parsed_output.reason
            })
            if parsed_output.is_affirmative == 'true':
                self.affirmative_score += 1
                self.affirmative_count += 1
            else:
                self.refuted_count += 1
        self.affirmative_score = round(self.affirmative_score / len(self.confirmation_qa_lists), 4)
        
        
    def evasiveness_eval(self):
        class OutputSchema(BaseModel):
            evasiveness: Literal['true', 'partially true', 'false'] = Field(..., description="Whether the provided response is evasive.")
            reason : str = Field(..., description="A brief explanation for the abstention decision.")
            evasiveness_type: Literal['none', 'refusal', 'lack info', 'asking back', 'unrelated'] = Field(..., description="The type of abstention (e.g., 'lack info', 'unknown', etc.).")
            
        qa_pairs = []
        for turn in self.state.history:
            if turn.type != 'repeat':
                for obs in turn.environment_observation:
                    if obs.observation_type == "interviewee_response":
                        qa_pairs.append({
                            "question": obs.response.question,
                            "answer": obs.response.content
                        })
        completion_kwargs_list = []
        for qa in qa_pairs:
            completion_kwargs = dict(
                model=self.agents['evaluator'].model,
                messages=[
                    {
                        "role": "system",
                        "content": self.evasive_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Question: {qa['question']}\nAnswer: {qa['answer']}"
                    }
                ],
                response_format=OutputSchema,
            )
            if self.agents['evaluator'].model.startswith("hosted_vllm/"):
                assert self.agents['evaluator'].port is not None, "Port must be specified for hosted_vllm models."    
                completion_kwargs['api_base'] = f"http://localhost:{self.agents['evaluator'].port}/v1"
            completion_kwargs_list.append(completion_kwargs)
            
        with ThreadPoolExecutor(max_workers=len(completion_kwargs_list)) as executor:
            results = list(executor.map(lambda kwargs: get_completion(**kwargs), completion_kwargs_list))
        for i, res in enumerate(results):
            self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
            output = res.choices[0].message.content if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            while output is None:
                logging.warning(f"Unexpected response for evasiveness eval: {output}. Retrying...")
                res = get_completion(**completion_kwargs_list[i])
                self.env_cost += completion_cost(res) if not self.agents['evaluator'].model.startswith("hosted_vllm/") else 0.0
                output = res.choices[0].message.content if res and res.choices and res.choices[0].message and res.choices[0].message.content else None
            try:
                parsed_output = OutputSchema.model_validate_json(output)
            except Exception as e:
                logging.error(f"Error parsing output for evasiveness eval: {e}")
                continue
            self.evasive_results.append({
                "question": qa_pairs[i]['question'],
                "response": qa_pairs[i]['answer'],
                "evasiveness": parsed_output.evasiveness,
                "reason": parsed_output.reason,
                "evasiveness_type": parsed_output.evasiveness_type
            })
            if parsed_output.evasiveness == 'true':
                self.evasive_score += 1
        self.evasive_score = round(self.evasive_score / len(qa_pairs), 4)
        
    
    def save_state(self, termination_status: str = "Successfully completed", reset_only: bool = False) -> dict:
        """save the current state to a json file"""
        agent_cost = sum(agent.cost for agent in self.agents.values())
        interviewee_cost = self.interviewee.calculate_cost()
        tool_costs = sum(tool.calculate_cost() for tool in self.tools.values())
        total_cost = agent_cost + interviewee_cost + self.env_cost + tool_costs

        final_result={
            "interview_date": self.cutoff_date,
            "agents_info":{agent_name: agent.model for agent_name, agent in self.agents.items()},
            "interviewee_info": {
                "name": self.interviewee.name,
                "baseline": self.interviewee.type,
            },
            "cost": {
                "agents_cost": agent_cost,
                "interviewee_cost": interviewee_cost,
                "environment_cost": self.env_cost,
                "tool_costs": {tool_name: tool.calculate_cost() for tool_name, tool in self.tools.items()},
                "total_cost": total_cost
            },
            "duration": f"{(time.time() - self.start_time)/60} min", # in minutes
            "termination_status": termination_status,
            "history": [obj.model_dump() for obj in self.state.history],
            "agent_memory": {
                agent_name: agent.memory for agent_name, agent in self.agents.items()
            },
        }
        if 'simulator_model' in self.interviewee.__dict__:
            final_result["interviewee_info"]["simulator_model"] = self.interviewee.simulator_model
            
        if not reset_only:
            self.evaluate()
            final_result["evaluation"] = {
                "consistency": {
                    "internal": {
                        "total": self.internal_count,
                        "conflict": self.internal_conflict,
                        "plausible": self.internal_plausible,
                        "conflict_verdicts": self.internal_conflict_verdicts
                    },
                    "external": {
                        "total": self.external_count,
                        "conflict": self.external_conflict,
                        "plausible": self.external_plausible,
                        "conflict_verdicts": self.external_conflict_verdicts
                    },
                    "first_conflict_turn": self.first_conflict_turn
                },
                "repeat": {
                    "repeat_score": self.repeat_score,
                    "details": self.repeat_results
                },
                "affirmative": {
                    "affirmative_score": self.affirmative_score,
                    "affirmative_count": self.affirmative_count,
                    "refuted_count": self.refuted_count,
                    "details": self.affirmative_results
                },
                "evasiveness": {
                    "evasive_score": self.evasive_score,
                    "details": self.evasive_results
                }
            }
        # write_json(final_result, path)
        
        logging.info("Cost: (agent):" + ", ".join([f"{agent_name}: ${agent.cost}" for agent_name, agent in self.agents.items()]))
        logging.info(f"Cost: (interviewee): ${interviewee_cost}")
        logging.info(f"Cost: (environment): ${self.env_cost}")
        logging.info(f"Total cost: ${final_result['cost']['total_cost']}, Duration: {final_result['duration']}")
        return final_result

if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging(log_to_file=True, process_name="test_env")
    load_dotenv()
    
    env = InterrogationEnv(
        model="gpt-5",
        baseline_name="characterai",
        character_id="6HhWfeDjetnxESEcThlBQtEUo0O8YHcXyHqCgN7b2hY", # example character id
        user_id=os.getenv('CAI_API_KEY'),
        name="Elon Musk",
        tools={
            "google_claim_search": GoogleClaimSearch(
                api_key=os.getenv('GOOGLE_CLAIM_SEARCH'),
                cx=os.getenv('GOOGLE_CX_ID'),
            ),
            "google_geocode_validate": GoogleGeocodeValidate(api_key=os.getenv('GOOGLE_GEOCODE'))
        },
    )
    state = env.reset()
    done = False
    while not done:
        state, done = env.step()
    state = env.finalize()
    env.save_state("interrogation_history.json")