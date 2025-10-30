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
        self.max_turns = max_turns + 1
        self.predefined_questions = read_json(question_path)
        self.instruction = open(instruction_path).read()
        self.state = State(current_turn=0, history=[])
        self.cutoff_date = None
        self.start_time = time.time()
        self.env_cost = 0.0

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
        # feed interview instruction to the interviewee
        logging.info(f"[INSTRUCTION] {self.instruction}")
        response = self.interviewee.get_response(self.instruction)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
        # run predefined questions
        for i, q in enumerate(self.predefined_questions):
            logging.info(f"[QUESTION] {q['question']}")
            response = self.interviewee.get_response(q['question'])
            logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
            if i == 0:
                # firt turn defines the cutoff date
                self.cutoff_date = response.content
                self.agents['questioner'].set_cutoff_date(self.cutoff_date)
                self.agents['web_search'].set_cutoff_date(self.cutoff_date)
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
            if i > 0:
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
            else:
                turn = Turn(type='get_to_know', agent_action=[action], environment_observation=[res_observation])
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
            filtered_actions = [action for action in web_search_actions if action and action.action_type == "tool_call"]
            
            if filtered_actions:
                with ThreadPoolExecutor(max_workers=len(filtered_actions)) as executor:
                    tool_outputs = list(executor.map(self.invoke_tool, filtered_actions))
                
                observation = Observation(
                    observation_type="tool_output",
                    tool_output=tool_outputs
                )

                for i, output in enumerate(tool_outputs):
                    sub_message = [
                        filtered_actions[i].tool_call.details,
                        {
                            "role": "tool",
                            "tool_call_id": output.tool_call_id,
                            "name": output.tool_name,
                            "content": str(output.output)
                        }
                    ]
                    """tool call 결과를 evaluator 메모리에 추가"""
                    self.agents['evaluator'].update_memory(**sub_message[0]) ####### 여기 #######
                    self.agents['evaluator'].update_memory(**sub_message[1]) ####### 여기 #######
                    messages = [
                        {
                            "role": "system",
                            "content": "Ask a single question to the interviewee to confirm or refute the information found in the web search results, e.g., \"Based on the search result, Google is ... Is the company what you meant? Please respond with 'yes' or 'no'.\""
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
                    logging.info(f"[CONFIRMATION QUESTION] {confirmation_question}")
                    response = self.interviewee.get_response(confirmation_question)
                    logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
                    
                    self.agents['evaluator'].update_memory(role="assistant", content=confirmation_question) ####### 여기 #######
                    self.agents['evaluator'].update_memory(role="user", content=response.content) ####### 여기 #######
                    
                    
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

        entity_action, observation, web_search_actions, verdict_action = self.consistency_check(
            interviewee_res.question,
            interviewee_res.content
        )

        self.state.current_turn += 1

        actions = [question_act, entity_action, *web_search_actions, *verdict_action] if web_search_actions else [question_act, entity_action, *verdict_action]
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
        }
        write_json(final_result, path)
        
        logging.info(f"Saving final result to {path}")
        logging.info(f"Total cost: ${final_result['total_cost']}, Duration: {final_result['duration']}")
        

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
            character_id="xUYbondr8di3UyRIbjdmmx1_BMLeUslf7KCA3N7h_hk", # example character id
            user_id=os.environ.get('CAI_API_KEY'),
            name="John Smith",
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
    env.save_state(f"data/prompt_engineering/questioner/questioner_test_history_{time.strftime('%Y%m%d_%H%M%S')}.json")