import time
from src.env.interviewee_simulator import IntervieweeSimulator
from typing import List, Dict, Any
from src.agents.base_agent import Agent
from src.agents.agent_factory import get_agent
from src.schemas import State, Action, Observation, Turn, ToolOutput, IntervieweeResponse
from src.tools.address_locator import GoogleGeocodeValidate
from src.tools.web_search import GoogleClaimSearch
from pydantic import BaseModel
from src.utils import read_json, write_json
import logging
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

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
        self.tools = tools
        if not agents:
            logging.warning("No agents provided. Initializing default agents.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            agents = {
                "questioner": get_agent("questioner", f"{project_root}/src/agents/prompts/examiner_prompt_2.txt"),
                "extractor": get_agent("claim_extractor", f"{project_root}/src/agents/prompts/claim_extractor_prompt.txt") if kwargs.get('use_claim_extractor', True) else get_agent("entity_extractor", f"{project_root}/src/agents/prompts/entity_extractor.txt"),
                "web_search": get_agent("web_search", f"{project_root}/src/agents/prompts/websearch_prompt.txt")
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

    def invoke_tool(self, action: Action) -> Observation | None:
        if action.action_type == "tool_call":
            tool_name = action.tool_call.tool_name
            if tool_name not in self.tools:
                logging.error(f"Tool {tool_name} not found.")
                return self.state, True
            
            # update questioner memory with tool calling details
            self.agents['questioner'].update_memory(
                **action.tool_call.details
            )
            
            tool = self.tools[tool_name]
            tool_output = tool.invoke(**action.tool_call.arguments)
            logging.info(f"[TOOL OUTPUT] {tool_name}: {tool_output[:100]}...") # print first 100 chars
            
            output = ToolOutput(
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
            observation = Observation(
                observation_type="interviewee_response",
                response=response
            )
            turn = Turn(type='get_to_know', agent_action=[action], environment_observation=[observation])
            self.state.history.append(turn)
            self.state.current_observation = observation
            self.state.current_turn += 1
            
            self.agents['questioner'].update_memory(role="assistant", content=q['question'])
            self.agents['questioner'].update_memory(role="user", content=response.content)
            
        return self.state

    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation"""
        interviewee_res = self.state.history[-1].environment_observation[-1].response
        message = f"Question:{interviewee_res.question}\nResponse: {interviewee_res.content}" # find interviewee_response (first index)
        if self.state.current_turn >= self.max_turns:
            logging.warning("Max turns reached. Please reset the environment.")
            return self.state, True, None
        
        # 1. Extractor first extracts the entity & claim to verify
        next_action = self.agents["extractor"].act(message)
        logging.info(f"[ACTION] Extractor: {next_action.action_type} - {next_action.content if next_action.content else next_action.target_agent}")
        if next_action.action_type == "next_agent":
            if next_action.target_agent != "questioner":
                logging.error("Extractor can only pass to Questioner.")
                return self.state, True
            logging.info("No entity or claim extracted. Passing to Questioner.")
            
        elif next_action.action_type == "respond": # should be respond with entity & claim
            if next_action.content is None:
                logging.error("Extractor must respond with entity & claim.")
                return self.state, True

            list_of_extractions = [ext.model_dump() for ext in next_action.content] # list of ExtractorResponse
            history = []
            for turn in self.state.history:
                # last element is interviewee response
                qa_pair = turn.environment_observation[-1].response
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
        # 3. Questioner formulates the next question
        # two scenarios: (1) from extractor directly (hence generating from interviewee's response directly), (2) from web search
        if 'observation' in locals():
            final_action = self.agents['questioner'].act(observation)
        else:
            final_action = self.agents['questioner'].act(
                self.state.history[-1].environment_observation[-1]
            )
        logging.info(f"[ACTION] Questioner: {final_action.action_type} - {final_action.content if final_action.content else final_action.tool_call.tool_name}")
        if final_action.action_type != "respond" or final_action.content is None:
            logging.error("Questioner must respond with a question.")
            return self.state, True
        # ask the question to the interviewee
        question = final_action.content
        logging.info(f"[QUESTION] {question}")
        response = self.interviewee.get_response(question)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
        
        # update state
        self.agents['questioner'].update_memory(role="user", content=response.content)
        self.state.current_turn += 1
        turn = Turn(
            type='main_interrogation',
            agent_action=[next_action, final_action],
            environment_observation=[
                Observation(
                    observation_type="interviewee_response",
                    response=response
                )
            ]
        )
        if 'filtered_actions' in locals():
            turn.agent_action.extend(filtered_actions) # include web search actions if any
        if 'observation' in locals():
            turn.environment_observation.insert(0, observation) # tool output should come before interviewee response, hence the last element is interviewee response
        self.state.history.append(turn)
        done = self.state.current_turn >= self.max_turns
        return self.state, done


    def finalize(self):
        """repeat stage: repeat the pre-defined questions to check for consistency"""
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
        return self.state
        
    
    def save_state(self, path: str):
        """save the current state to a json file"""
        final_result={
            "agents_info":{agent_name: agent.model for agent_name, agent in self.agents.items()},
            "interviewee_info": {
                "name": self.interviewee.name,
                "baseline": self.interviewee.type,
            },
            "duration": f"{(time.time() - self.start_time)/60} min", # in minutes
            "history": [obj.model_dump() for obj in self.state.history],
            "agent_memory": {
                agent_name: agent.memory for agent_name, agent in self.agents.items()
            }
        }
        
        write_json(final_result, path)


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging(log_to_file=True, process_name="test_env")
    load_dotenv()
    
    env = InterrogationEnv(
        baseline_name="characterai",
        character_id="6HhWfeDjetnxESEcThlBQtEUo0O8YHcXyHqCgN7b2hY", # example character id
        user_id="YOUR_USER_ID",
        name="Elon Musk",
        tools={
            "google_claim_search": GoogleClaimSearch(
                api_key='YOUR_API_KEY',
                cx='YOUR_CX'
            ),
            "google_geocode_validate": GoogleGeocodeValidate(api_key='YOUR_API_KEY')
        },
        max_turns=30
    )
    state = env.reset()
    done = False
    while not done:
        state, done = env.step()
    state = env.finalize()
    env.save_state("interrogation_history.json")