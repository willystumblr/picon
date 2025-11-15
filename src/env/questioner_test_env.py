import time
from src.env.interviewee_simulator import IntervieweeSimulator
from typing import List, Dict, Any
from src.agents.base_agent import Agent
from src.agents.agent_factory import get_agent
from src.schemas import State, Action, Observation, Turn, ToolOutput, IntervieweeResponse
from src.tools.address_locator import GoogleGeocodeValidate
from src.tools.web_search import GoogleClaimSearch
from pydantic import BaseModel
from src.utils import read_json, write_json, get_completion
import logging
from litellm.cost_calculator import completion_cost
import os
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import random

class QuestionerTestEnv:
    def __init__(
        self, 
        model, 
        agents: Dict[str, Agent] = {},
        baseline_name: str = "characterai",
        tools: List[Dict[str, Any]] = [],
        max_turns: int = 20,
        question_path: str = "src/env/wvs_orthogonal_questions.json",
        instruction_path: str = "src/env/interrogation_instruct.txt",
        **kwargs
        ):
        self.tools = tools
        self.model = model
        if not agents:
            logging.warning("No agents provided. Initializing default agents.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            agents = {
                "questioner": get_agent("questioner", f"{project_root}/src/agents/prompts/questioner.txt", model=model),
                "extractor": get_agent("entity_extractor", f"{project_root}/src/agents/prompts/entity_extractor.txt", model=model),
                "web_search": get_agent("web_search", f"{project_root}/src/agents/prompts/websearch_prompt.txt", model=model),
                "evaluator": get_agent("evaluator", f"{project_root}/src/agents/prompts/evaluator_prompt.txt", model=model),
            }
        self.agents = agents
        if 'web_search' in self.agents and not self.tools:
            logging.warning("No tools provided for web search agent.")
        if 'web_search' in self.agents and not self.agents['web_search'].tools:
            logging.info("Setting web search agent tools from environment.")
            self.agents['web_search'].tools = [tool.get_info() for tool in self.tools.values()]
        
        self.interviewee = IntervieweeSimulator(
            baseline_name=baseline_name,
            **kwargs # simulator specific args (character_id, user_id, name for characterai; model_path, persona, profile for opencharacter; name for human_simulacra)
        )
        self.max_turns = max_turns
        self.predefined_questions = random.sample(read_json(question_path), k=5)
        self.instruction = open(instruction_path).read()
        self.state = State(current_turn=0, history=[])
        self.cutoff_date = time.strftime("%B %d, %Y") # default to current date
        self.start_time = None
        self.env_cost = 0.0
        
        self.agents['questioner'].set_cutoff_date(self.cutoff_date)
        self.agents['web_search'].set_cutoff_date(self.cutoff_date)
        self.agents['evaluator'].set_cutoff_date(self.cutoff_date)

        # repeat
        self.repeat_score = 0
        self.repeat_results = []
        
        self.human_interviewer = kwargs.get("human_interviewer", False)

    def invoke_tool(self, action: Action) -> Observation | None:
        if action.action_type == "tool_call":
            tool_name = action.tool_call.tool_name
            if tool_name not in self.tools:
                logging.error(f"Tool {tool_name} not found.")
                return self.state, True
            
            # update questioner memory with tool calling details
            # self.agents['questioner'].update_memory(
            #     **action.tool_call.details
            # )
            
            tool = self.tools[tool_name]
            tool_output = tool.invoke(**action.tool_call.arguments)
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
            logging.info(f"[QUESTION {self.state.current_turn}] {q['question']}")
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
            
            entity_action, web_observation, web_search_actions = self.consistency_check(
                question=q['question'],
                answer=response.content
            )
            # for verdict_action in verdict_actions:
                # self.agents['questioner'].update_memory(**{"role":"assistant", "content": str(verdict_action.content)}) ####### 여기 #######
            if web_search_actions:
                turn = Turn(type='get_to_know', agent_action=[entity_action, *web_search_actions], environment_observation=[res_observation, web_observation])
            else:
                turn = Turn(type='get_to_know', agent_action=[entity_action], environment_observation=[res_observation])
            
            self.state.history.append(turn)

        return self.state

    def check_external(self, message : str) -> bool:
        # 1. Extractor first extracts the entity & claim to verify
        next_action = self.agents["extractor"].act(message)
        logging.info(f"[ACTION] Extractor: {next_action.action_type} - {next_action.content if next_action.content else next_action.target_agent}")
        if next_action.action_type == "next_agent":
            if next_action.target_agent != "questioner":
                logging.error("Extractor can only pass to Questioner.")
                return self.state, True
            logging.info("No entity or claim extracted. Passing to Questioner.")
            observation = None
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
                
                observation = Observation(
                    observation_type="tool_output",
                    tool_output=tool_outputs
                )
                prev_conf_qa = []
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
                        }
                    ]
                    """tool call 결과를 evaluator 메모리에 추가"""
                    self.agents['evaluator'].update_memory(**sub_message[1]) ####### 여기 #######
                    self.agents['evaluator'].update_memory(**sub_message[2]) ####### 여기 #######
                    if prev_conf_qa:
                        sub_message = sub_message + prev_conf_qa
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Ask a short, concise \"confirm/refute\" question if the entity that the interviewee mentioned refers to the information found in the web search results. You may provide a brief explanation about the entity based on the search results. "
                                "Assume that no further search is available beyond the provided search results. "
                                "If search results are incomplete due to search failure or error, ask a generic confirmation question about the entity. "
                                "If the search results are duplicate or redundant with previous questions, do not ask a new question; instead, responde with a single word-SKIP."
                                "Generate a single question without any additional explanation. "
                            )
                        },
                    ]
                    messages.extend(sub_message)
                    res = get_completion(
                        model=self.model,
                        messages=messages,
                        reasoning_effort="low",
                    )
                    self.env_cost += completion_cost(res)
                    confirmation_question = res.choices[0].message.content.strip()
                    if not confirmation_question or confirmation_question.upper() == "SKIP":
                        logging.info("No new confirmation question generated. Skipping.")
                        continue
                    else:
                        logging.info(f"[CONFIRMATION QUESTION] {confirmation_question}")
                        response = self.interviewee.get_response(confirmation_question)
                        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
                        
                        self.agents['evaluator'].update_memory(role="assistant", content=confirmation_question) ####### 여기 #######
                        self.agents['evaluator'].update_memory(role="user", content=response.content) ####### 여기 #######
                        self.agents['questioner'].update_memory(role="assistant", content=confirmation_question) ####### 여기 #######
                        self.agents['questioner'].update_memory(role="user", content=response.content) ####### 여기 #######

                        prev_conf_qa.append({
                            "role": "assistant",
                            "content": confirmation_question
                        })
                        prev_conf_qa.append({
                            "role": "user",
                            "content": response.content
                        })
                    
            else:
                observation = None
                filtered_actions = []
        return next_action, observation, filtered_actions

    def consistency_check(self, question: str, answer : str):
        message = f"Question:{question}\nResponse: {answer}" # find interviewee_response (first index)
        
        next_action, observation, filtered_actions = self.check_external(message)
        
        return next_action, observation, filtered_actions
    
    
    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation"""
        logging.info(f"--- Turn {self.state.current_turn} ---")
        if self.state.current_turn >= self.max_turns:
            logging.warning("Max turns reached. Please reset the environment.")
            return self.state, True
        verdict = self.state.history[-1].agent_action[-1].content if self.state.history else None
        question_act = self.agents['questioner'].act(verdict=verdict)
        logging.info(f"[ACTION] Questioner: {question_act.action_type} - {question_act.content if question_act.content else question_act.tool_call.tool_name}")
        self.agents['evaluator'].update_memory(role="assistant", content=question_act.content)
        interviewee_res = self.interviewee.get_response(question_act.content)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {interviewee_res.content}")
        
        self.agents['questioner'].update_memory(role="user", content=interviewee_res.content) 
        self.agents['evaluator'].update_memory(role="user", content=interviewee_res.content)

        entity_action, observation, web_search_actions = self.consistency_check(
            interviewee_res.question,
            interviewee_res.content
        )

        self.state.current_turn += 1

        actions = [question_act, entity_action, *web_search_actions] if web_search_actions else [question_act, entity_action]
        res_ob = Observation(observation_type="interviewee_response", response=interviewee_res)
        observations = [res_ob, observation] if observation else [res_ob]
        turn = Turn(
            type='main_interrogation',
            agent_action=actions,
            environment_observation=observations
        )
        
        self.state.history.append(turn)

        return self.state, False
        
    def save_state(self, path: str, termination_status: str = "Successfully completed"):
        """save the current state to a json file"""
        final_result={
            "agents_info":{agent_name: agent.model for agent_name, agent in self.agents.items()},
            "interviewee_info": {
                "name": self.interviewee.name,
                "baseline": self.interviewee.type,
            },
            "total_cost": sum(agent.cost for agent in self.agents.values()) + self.interviewee.cost + self.env_cost,
            "duration": f"{(time.time() - self.start_time)/60} min", # in minutes
            "termination_status": termination_status,
            "history": [obj.model_dump() for obj in self.state.history],
            "agent_memory": {
                agent_name: agent.memory for agent_name, agent in self.agents.items()
            },
            "repeat_score":{
                "repeat_score": self.repeat_score,
                "repeat_results": self.repeat_results
            }
        }
        write_json(final_result, path)
        
        logging.info(f"Saving final result to {path}")
        logging.info(f"Total cost: ${final_result['total_cost']}, Duration: {final_result['duration']}")

    def finalize(self):
        """repeat stage: repeat the pre-defined questions to check for consistency"""
        
        REPEAT_PROMPT = """You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""
        
        for i, q in enumerate(self.predefined_questions):
            logging.info(f"[REPEAT QUESTION] Just to clarify, {q['question']}")
            response = self.interviewee.get_response(f"Just to clarify, {q['question']}")
            logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
            if i == 0:
                # first turn defines the cutoff date
                continue
            # update state
            action = Action(action_type="respond", content=q['question'])
            observation = Observation(
                observation_type="interviewee_response",
                response=response
            )
            turn = Turn(type='repeat', agent_action=[action], environment_observation=[observation])
            self.state.history.append(turn)
            
            inital_response = self.state.history[i].environment_observation[0].response.content
            while True:
                res = get_completion(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": REPEAT_PROMPT},
                        {"role": "user", "content": f"Question: {q['question']}\n\nResponse 1: {inital_response}\nResponse 2: {response.content}"}
                    ],
                    reasoning_effort="low",
                )
                self.env_cost += completion_cost(res)
                judge = res.choices[0].message.content.strip() if res and res.choices and res.choices[0].message and res.choices[0].message.content else "FALSE"
                if judge in ["TRUE", "FALSE"]:
                    break
                logging.warning(f"Unexpected response for repeat score: {judge}. Retrying...")
            
            self.repeat_results.append({
                "question": f"Just to clarify, {q['question']}",
                "original_response": inital_response,
                "repeated_response": response.content,
                "is_repeat": judge
            })
            self.repeat_score += (judge=='TRUE')
        self.repeat_score = round(self.repeat_score / len(self.predefined_questions), 4)
        return self.state

if __name__ == "__main__":
    from src.utils import setup_logging
    from argparse import ArgumentParser
    setup_logging(log_to_file=True, process_name="test_env")
    load_dotenv()

    parser = ArgumentParser(description="Questioner Test Environment")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use")
    parser.add_argument("--nhd_model", type=str, default="gpt-5", help="NHD model to use")
    parser.add_argument("--max_turns", type=int, default=30, help="Maximum number of turns")
    parser.add_argument("--human_interviewer", action="store_true", help="Use human simulacra as interviewer")
    parser.add_argument("--human_interviewee", action="store_true", help="Use human simulacra as interviewee")
    args = parser.parse_args()
    
    if args.human_interviewee:
        env = QuestionerTestEnv(
            model=args.model,
            baseline_name="human_interview",
            nhd_model=args.model,
            max_turns=args.max_turns,
            human_interviewer=args.human_interviewer,
            tools={
                "google_claim_search": GoogleClaimSearch(
                    api_key=os.environ.get('GOOGLE_CLAIM_SEARCH'),
                    cx=os.environ.get('GOOGLE_CX_ID')
                ),
                "google_geocode_validate": GoogleGeocodeValidate(api_key=os.environ.get('GOOGLE_GEOCODE'))
            },
        )

    else:
        env = QuestionerTestEnv(
            model=args.model,
            baseline_name="characterai",
            character_id="6HhWfeDjetnxESEcThlBQtEUo0O8YHcXyHqCgN7b2hY", # example character id
            user_id=os.environ.get('CAI_API_KEY'),
            name="Elon Musk",
            tools={
                "google_claim_search": GoogleClaimSearch(
                    api_key=os.environ.get('GOOGLE_CLAIM_SEARCH'),
                    cx=os.environ.get('GOOGLE_CX_ID')
                ),
                "google_geocode_validate": GoogleGeocodeValidate(api_key=os.environ.get('GOOGLE_GEOCODE'))
            },
            max_turns=args.max_turns,
            nhd_model=args.model,
            human_interviewer=args.human_interviewer
        )
    state = env.reset()
    done = False
    while not done:
        state, done = env.step()
    env.finalize()
    env.save_state(f"data/prompt_engineering/questioner/questioner_test_history_{time.strftime('%Y%m%d_%H%M%S')}.json")