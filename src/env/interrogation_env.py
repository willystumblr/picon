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
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

CONFLICT_PAIR_PROMPT = open(f"{project_root}/src/env/conflict_detection_prompt.txt").read()

REPEAT_PROMPT = """You will be given a single question and two corresponding answers. Determine whether the two answers are essentially the same in meaning.
If they are, output TRUE. If they are not, output FALSE.
Do not output any additional explanation or text."""


class InterrogationEnv:
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
            agents = {
                "questioner": get_agent("questioner", f"{project_root}/src/agents/prompts/examiner_prompt_2.txt"),
                "extractor": get_agent("claim_extractor", f"{project_root}/src/agents/prompts/claim_extractor_prompt.txt") if kwargs.get('use_claim_extractor', True) else get_agent("entity_extractor", f"{project_root}/src/agents/prompts/entity_extractor.txt"),
                "web_search": get_agent("web_search", f"{project_root}/src/agents/prompts/websearch_prompt.txt"),
                "kg_agent": get_agent("kg_agent", f"{project_root}/src/agents/prompts/kg_agent_prompt.txt", model=model)
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

        # internal 
        self.internal_conflict_pairs_cnt = 0
        self.internal_conflict_pairs = []
        self.total_pairs_evaluated = 0
        self.first_conflict_turn = False
        
        # external
        self.con_cnt = 0
        self.incon_cnt = 0
        self.unknown_cnt = 0
        self.total_confirmations = 0
        self.confirmed_answers = 0
        self.confirmed_results = []
        
        # repeat
        self.repeat_score = 0
        self.repeat_results = []
    

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
        qa_history = ""
        for i, q in enumerate(self.predefined_questions):
            logging.info(f"[QUESTION] {q['question']}")
            response = self.interviewee.get_response(q['question'])
            logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
            if i == 0:
                # firt turn defines the cutoff date
                self.cutoff_date = response.content
                self.agents['questioner'].set_cutoff_date(self.cutoff_date)
                self.agents['web_search'].set_cutoff_date(self.cutoff_date)
                init_triplet = self.agents['kg_agent'].act(f"The interviewee's cutoff date information {self.cutoff_date}.")
                logging.info(f"[ACTION] KG Agent: {init_triplet.action_type} - {init_triplet.content if init_triplet.content is not None else init_triplet.tool_call.tool_name}")
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
            qa_history += f"Q: {q['question']}\nA: {response.content}\n"
            if i > 0:
                next_action, observation, filtered_actions, confirmed_results, conflict_pairs = asyncio.run(self.consistency_check(
                    question=q['question'],
                    answer=response.content
                ))
                
        self.agents['extractor'].memory[0]['content'] = self.agents['extractor'].memory[0]['content']+f"QA History: {qa_history}"
        question = self.agents['questioner'].act() ####### 여기 #######
        logging.info(f"[ACTION] Questioner: {question.action_type} - {question.content if question.content else question.tool_call.tool_name}")
        logging.info(f"[FIRST QUESTION] {question.content}")
        response = self.interviewee.get_response(question.content)
        logging.info(f"[RESPONSE] {self.interviewee.name}: {response.content}")
        self.agents['questioner'].update_memory(role="user", content=response.content)
        self.state.current_turn += 1
        turn = Turn(
            type='main_interrogation',
            agent_action=[question],
            environment_observation=[
                Observation(
                    observation_type="interviewee_response",
                    response=response
                )
            ]
        )
        self.state.history.append(turn)
        return self.state

    async def check_external(self, message : str) -> bool:
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
            confirmed_results = []
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

                # confirmation questions
                confirmed_results = []
                for i, output in enumerate(tool_outputs):
                    messages = [
                        {
                            "role": "system",
                            "content": "Ask a single question to the interviewee to confirm or refute the information found in the web search results, e.g., \"Based on the search result, Google is ... Is the company what you meant? Please respond with 'yes' or 'no'.\""
                        },
                        filtered_actions[i].tool_call.details,
                        {
                            "role": "tool",
                            "tool_call_id": output.tool_call_id,
                            "name": output.tool_name,
                            "content": str(output.output)
                        }
                    ]
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
                    content = f"Question: {confirmation_question}\nInterviewee's Response: {response.content}"
                    res = get_completion(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "Determine if the interviewee’s response confirms that the web search results match what they said. Respond with 'yes' if confirmed, 'no' otherwise."
                            },
                            {
                                "role": "user",
                                "content": content
                            }
                        ],
                        reasoning_effort="low",
                    )
                    self.env_cost += completion_cost(res)
                    confirmed = res.choices[0].message.content.strip().lower()
                    confirmed_result = {
                        "claim": filtered_actions[i].tool_call.arguments.get('claim', ''),
                        "original_qa": message,
                        "content": str(output.output),
                        "confirmation_qa": content,
                    }
                    self.total_confirmations += 1
                    if confirmed == 'yes':
                        self.confirmed_answers += 1
                        confirmed_result["is_confirmed"] = True
                    else:
                        self.unknown_cnt += 1
                        confirmed_result['confirmed_message'] = confirmed
                        confirmed_result["is_confirmed"] = False
                    confirmed_results.append(confirmed_result)
            else:
                observation = None
                filtered_actions = []
                confirmed_results = []
        return next_action, observation, filtered_actions, confirmed_results

    async def check_internal(self, question: str, answer : str) -> bool:
        conflict_count = 0
        # pair_format = "Question 1: {question_1}\nResponse 1: {response_1}\n\nQuestion 2: {question_2}\nResponse 2: {response_2}"
        pair_format = "Triplet 1: {triplet_1} \n\nTriplet 2: {triplet_2}"
        # prev_qas = [(turn.environment_observation[-1].response.question, turn.environment_observation[-1].response.content) for turn in self.state.history[:-1]]
        triplets_res = self.agents['kg_agent'].act(f"Question: {question}\nResponse: {answer}")
        triplets_2 = triplets_res.content
        if not triplets_2:
            logging.warning(f"[ACTION] KG Agent: {triplets_res.action_type} - No triplets found")
            return conflict_count, []
        logging.info(f"[ACTION] KG Agent: {triplets_res.action_type} - {triplets_res.content if triplets_res.content is not None else triplets_res.tool_call.tool_name}")
        prev_triplets = self.agents['kg_agent'].kg[:-len(triplets_2)] # all previously extracted triplets
        content_list = []
        for triplet_2 in triplets_2:
            content_list.extend([pair_format.format(
                triplet_1=triplet, triplet_2=triplet_2
            ) for triplet in prev_triplets])
        # Create combinations within the response triplets_2:
        combinations_within_response = combinations(triplets_2, 2)
        for triplet_1, triplet_2 in combinations_within_response:
            content_list.append(pair_format.format(
                triplet_1=triplet_1, triplet_2=triplet_2
            ))
        messages_list = [[
            {
                "role": "system",
                "content": CONFLICT_PAIR_PROMPT
            },
            {
                "role": "user",
                "content": content
            }
        ] for content in content_list]
        conflict_pairs = []
        logging.info("Using sequential processing for conflict pair evaluation.")
        logging.info(f"Total pairs to evaluate: {len(messages_list)}")
        logging.info(f"Using model: {self.model}")
        logging.info("This may take a while...")
        with ThreadPoolExecutor(max_workers=32) as executor: # using tqdm for progress bar
            results = list(tqdm(executor.map(lambda msg: get_completion(model=self.model, messages=msg), messages_list), total=len(messages_list), desc="Evaluating Conflict Pairs"))
            if isinstance(self.first_conflict_turn, bool) and not self.first_conflict_turn and any(res and res.choices and res.choices[0].message and res.choices[0].message.content and res.choices[0].message.content.strip() == 'conflict' for res in results):
                self.first_conflict_turn = self.state.current_turn
                logging.info(f"First conflict detected at turn {self.state.current_turn}.")
        self.total_pairs_evaluated += len(results)
        for i, res in enumerate(results):
            if not res or not res.choices or not res.choices[0].message or not res.choices[0].message.content or res.choices[0].message.content.strip() not in ['plausible', 'conflict']:
                logging.warning(f"Unexpected response: {res}")
                continue
            self.env_cost += completion_cost(res)
            if res.choices[0].message.content.strip() == "conflict":
                conflict_count += 1
                conflict_pairs.append(content_list[i])

        return conflict_count, conflict_pairs
    
    async def consistency_check(self, question: str, answer : str):
        message = f"Question:{question}\nResponse: {answer}" # find interviewee_response (first index)
        
        # internal consistency check
        conflict_count, conflict_pairs = await self.check_internal(question, answer)
        self.internal_conflict_pairs_cnt += conflict_count
        self.internal_conflict_pairs.extend(conflict_pairs)

        next_action, observation, filtered_actions, confirmed_results = await self.check_external(message)
        self.confirmed_results.extend(confirmed_results)
        
        return next_action, observation, filtered_actions, confirmed_results, conflict_pairs
    
    
    def step(self): # Interviewee's response -> Extractor -> WebSearch (optional) -> Questioner -> Interviewee
        """run one turn of the interrogation"""
        interviewee_res = self.state.history[-1].environment_observation[-1].response
        

        next_action, observation, filtered_actions, confirmed_results, conflict_pairs = asyncio.run(self.consistency_check(
            interviewee_res.question,
            interviewee_res.content
        ))
        
        if self.state.current_turn >= self.max_turns:
            logging.warning("Max turns reached. Please reset the environment.")
            return self.state, True
        
        # 3. Questioner formulates the next question
        # two scenarios: (1) from extractor directly (hence generating from interviewee's response directly), (2) from web search
        
        final_action = self.agents['questioner'].act() ####### 여기 #######
    
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
        if filtered_actions:
            turn.agent_action.extend(filtered_actions) # include web search actions if any
        if observation:
            turn.environment_observation.insert(0, observation) # tool output should come before interviewee response, hence the last element is interviewee response
        self.state.history.append(turn)

        return self.state, False


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
            "external_consistency": {
                "confirmed_rate": self.confirmed_answers / self.total_confirmations if self.total_confirmations > 0 else 0,
                "confirmed_results": self.confirmed_results
            },
            "internal_consistency": {
                "first_conflict_turn": self.first_conflict_turn,
                "conflict_pairs_count": self.internal_conflict_pairs_cnt,
                "conflict_pairs": self.internal_conflict_pairs,
                "total_pairs_evaluated": self.total_pairs_evaluated,
                "conflict_rate": self.internal_conflict_pairs_cnt / self.total_pairs_evaluated if self.total_pairs_evaluated > 0 else 0
            },
            "repeat":{
                "repeat_score": self.repeat_score,
                "repeat_results": self.repeat_results
            }
        }
        write_json(final_result, path)
        
        logging.info(f"Saving final result to {path}")
        logging.info(f"Total cost: ${final_result['total_cost']}, Duration: {final_result['duration']}")
        logging.info(f"First conflict turn: {final_result['internal_consistency']['first_conflict_turn']}")
        logging.info(f"External confirm rate: {final_result['external_consistency']['confirmed_rate']}")
        logging.info(f"Internal consistency: {final_result['internal_consistency']['conflict_rate']}")
        

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