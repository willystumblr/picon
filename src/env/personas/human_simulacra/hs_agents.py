import os
import argparse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback
import json
import re
import sys
import logging
from colorama import Fore, Style
from dotenv import load_dotenv
from tiktoken import encoding_for_model
from litellm import get_max_tokens

load_dotenv()  # Load environment variables from .env file
BASE_DIR = os.path.dirname(__file__)
PROMPT_DIR = f"{BASE_DIR}/prompts"
BASE_URL = "https://api.openai.com/v1"
CHARACTERS = ["Mary Jones", "Haley Collins", "Sara Ochoa", "James Jones", "Tami Clark", "Michael Miller", "Kevin Kelly", "Erica Walker", "Leslie Nichols", "Robert Scott", "Marsh Zhaleh"]
INTRODUCTIONS_PATH = f"{BASE_DIR}/Characters/character_introductions.json"
STORY_DIR = f"{BASE_DIR}/Characters/Stories"
MEMORY_DIR = f"{BASE_DIR}/Characters/Memories"



def get_context_limit(model: str, provider: str | None = None) -> int | None:
    """
    provider 예: "openai", "anthropic", "gemini", "bedrock", "azure"
    provider가 None이면 자동 추론
    """
    try:
        if provider:
            return get_max_tokens(model=model, custom_llm_provider=provider)
        return get_max_tokens(model=model)
    except Exception:
        return None


class Memory_agent:
    # The Memory agent is responsible for the following functions:
    # (1) Add long-term memory, stored in Long_memory.json;
    # (2) Add short-term memory, stored in Short_memory.txt;
    # (3) Retrieval: Retrieve the most relevant memory summaries from Index.json using LLM based on the query, then find the corresponding memories in Long_memory.json.
    def __init__(self, character_name, model: str="gpt-4.1-mini-2025-04-14", temperature=0.0, api_base=BASE_URL, api_key=os.environ['OPENAI_API_KEY']):
        
        self.name = character_name
        self.temperature = temperature
        self.api_base = api_base
        self.api_key = api_key
        self.path = os.path.join(MEMORY_DIR, character_name)
        self.cost = 0.0
        self.sum = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=model,
            temperature = self.temperature
        )
        self.retrieval = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=model,
            temperature = self.temperature
        )
        self.system_prompt = open(os.path.join(PROMPT_DIR, "memory_agent_system_prompt_template.txt")).read().format(
            character_name = self.name
        )

    def Summary(self, memory_chunk, emotion):
        # Generate a summary of a memory chunk for subsequent memory retrieval.
        system_prompt = open(os.path.join(PROMPT_DIR, "memory_summary_system_prompt_template.txt")).read()
        user_prompt = open(os.path.join(PROMPT_DIR, "memory_summary_user_prompt_template.txt")).read().format(
            character_name = self.name,
            memory_chunk = memory_chunk,
            emotion = emotion
        )
        messages = []
        messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=user_prompt))

        # Generate summary
        with get_openai_callback() as cb:
            summary = self.sum(messages).content
            self.cost += cb.total_cost
        return summary

    def Save_index_file(self, index):
        # Construct the index file (index.json) with the structure: {"num1": "Memory_Summary1", "num2": "Memory_Summary2"....}
        file_path = os.path.join(self.path, "index.json")
        if not os.path.exists(file_path):  # 检查文件是否存在
            with open(file_path, 'w') as file:
                json.dump(index, file)  # 写入内容

            print(f"File {file_path} created successfully!")
        else:
            print(f"File {file_path} already exists")
            with open(file_path, 'w') as file:
                json.dump(index, file)

    def Save_long_memory(self, memory):
        # Save long-term memory to long_memory.json
        file_path = os.path.join(self.path, "long_memory.json")  # 连接文件夹路径和文件名
        if not os.path.exists(self.path):  # 检查文件是否存在
            with open(file_path, 'w') as file:
                json.dump(memory, file)  # 写入内容
                
            print(f"File {file_path} created successfully!")
        else:
            print(f"File {file_path} already exists, replace it with a new one!")
            with open(file_path, 'w') as file:
                json.dump(memory, file)     
    
    # def Add_short_memory(self, memory):
    #     # Add short-term memory to short_memory.txt
    #     file_path = os.path.join(self.path, "short_memory.txt")  # 连接文件夹路径和文件名
    #     if not os.path.exists(self.path):  # 检查文件是否存在
    #         with open(file_path, 'w') as file:
    #             file.write(memory)  # 写入内容
    #             file.write('\n')
    #         print(f"File {file_path} created successfully!")
    #     else:
    #         print(f"File {file_path} already exists")
    #         with open(file_path, 'a') as file:
    #             file.write(memory)
    #             file.write('\n')

    def Search(self, result_list):
        # Retrieve relevant memories from long_memory.json based on result_list
        file_path = os.path.join(self.path, "long_memory.json")
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        Search_resurt = {}
        Memory_list = []
        key_list = result_list
        for key in key_list:
        # 检查键是否存在于JSON数据中
            if key in json_data:
                # 获取对应键的"Memeory"内容
                memory = json_data[key]["Memory"]
                # 添加到结果列表中
                Memory_list.append(memory)
        num = 0
        limit = 1 ## one most related memory
        for memory in Memory_list:
            num += 1
            if num > limit:
                break
            string_num = str(num).zfill(3)
            Search_resurt[string_num] = memory
        return Search_resurt
    

    def Memory_Retrieval(self, Query):
        # Perform memory retrieval based on LLM, memories are stored in Long_memory.json
        # Memory structure: {"num1": {"Memory_Summary": "xxx", "Memory": {"Memory Content": "xxx", "Thinking": "xxx", "Emotion": "xxx"}}, "num2": {"Memory_Summary": "xxx", "Memory": {"Memory Content": "xxx", "Thinking": "xxx", "Emotion": "xxx"}}, ... }
        # The retrieval process involves LLM determining the most relevant Memory_Summaries for the query, then aggregating the corresponding Memories as the retrieval result.
        messages1 = []
        messages1.append(SystemMessage(content=self.system_prompt))
        messages2 = []
        messages2.append(SystemMessage(content=self.system_prompt))
        index_file_path = os.path.join(self.path, "index.json")
        # Index file name is Index.json, structure: {"num1": "Memory_Summary1", "num2": "Memory_Summary2"....}
        # The retrieval result returns memory segment numbers (num)
        with open(index_file_path, 'r') as file:
            index = json.load(file)
        
        keys = list(index.keys())

        # Split index file into two parts due to potential context length limit, calculate split point
        split_point = len(keys) // 2

       # Use slicing to split the key list
        keys1 = keys[:split_point]
        keys2 = keys[split_point:]

        index1 = {key: index[key] for key in keys1}
        index2 = {key: index[key] for key in keys2}

        user_prompt = open(os.path.join(PROMPT_DIR, "memory_agent_user_prompt_template.txt")).read().format(
            index = index1,
            query = Query
        )
        messages1.append(HumanMessage(content = user_prompt))
        
        user_prompt = open(os.path.join(PROMPT_DIR, "memory_agent_user_prompt_template.txt")).read().format(
            index = index2,
            query = Query
        )
        messages2.append(HumanMessage(content=user_prompt))
        with get_openai_callback() as cb:
            ans = self.retrieval.invoke(messages1).content
            ans += self.retrieval.invoke(messages2).content
            self.cost += cb.total_cost
        pattern = r'"\d{3}"'  
        matches = re.findall(pattern, ans)  
        result_list = list(set([match.strip('"') for match in matches]))
        # Find corresponding memories based on retrieval results
        Retrieval_result = self.Search(result_list)
        return Retrieval_result
    
class Thinking_agent:
    # The Thinking_agent class is responsible for the following functions:
    # (1) Analyze the thinking process of the character based on the query;
    # (2) Construct "Memory Content" and "thinking" based on a segment of the Life_story.
    def __init__(self, character_infos, character_name, character_biography, personality_traits, model: str="gpt-4.1-mini-2025-04-14", temperature=0.0, api_base=BASE_URL, api_key=os.environ['OPENAI_API_KEY']):
        self.api_base = api_base
        self.api_key = api_key
        self.infos = character_infos
        self.name = character_name
        self.biography = character_biography
        self.personality_traits = personality_traits,
        self.temperature = temperature
        self.think = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=model,
            temperature = self.temperature
        )
        self.cost = 0.0
    
    def Memory_construction(self, LifeStory_chunk):
        # Construct "Memory Content" and "thinking" based on a segment of the Life_story
        messages = []
        sys_prompt = open(os.path.join(PROMPT_DIR, "memory_content_construction_system_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        user_prompt = open(os.path.join(PROMPT_DIR, "memory_content_construction_user_prompt_template.txt")).read().format(
            chunk = LifeStory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate memory
        with get_openai_callback() as cb:
            ans = self.think.invoke(messages)
            self.cost += cb.total_cost
        return ans.content
    
    def Thinking_Memory_construction(self, memory_chunk):
        # Generate the character's thinking about a memory chunk
        messages = []
        sys_prompt = open(os.path.join(PROMPT_DIR, "thinking_memory_construction_system_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        
        user_prompt = open(os.path.join(PROMPT_DIR, "thinking_memory_construction_user_prompt_template.txt")).read().format(
            chunk = memory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate thinking about the memory chunk
        with get_openai_callback() as cb:
            ans = self.think.invoke(messages)
            self.cost += cb.total_cost
        return ans.content
    
    def Thinking_analysis(self, query):
        # Analyze the character's current thinking process based on the query
        messages = []
        sys_prompt = open(os.path.join(PROMPT_DIR, "generate_personal_think_system_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            character_biography = self.biography
        )
        user_prompt = open(os.path.join(PROMPT_DIR, "generate_personal_think_user_prompt_template.txt")).read().format(
            query = query
        )

        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        with get_openai_callback() as cb:
            Thinking_result = self.think.invoke(messages)
            self.cost += cb.total_cost
            
        return Thinking_result.content

class Emotion_agent:
    # The Emotion_agent class is responsible for the following functions:
    # (1) Analyze the character's current emotional state based on the query;
    # (2) Construct "Emotion" based on a segment of the Life_story.
    def __init__(self, character_infos, character_name, personality_traits, model:str, temperature=0.0, api_base=BASE_URL, api_key=os.environ['OPENAI_API_KEY']):
        self.api_base = api_base
        self.api_key = api_key
        self.infos = character_infos
        self.name = character_name
        self.personality_traits = personality_traits
        self.temperature = temperature
        self.emotion = ChatOpenAI(
            openai_api_base=self.api_base,
            openai_api_key=self.api_key,
            model=model,
            temperature = self.temperature
        )
        self.cost = 0.0

    def Memory_construction(self, LifeStory_chunk):
        # Construct "Emotion Memory" based on a segment of the Life_story
        messages = []
        sys_prompt = open(os.path.join(PROMPT_DIR, "emotional_memory_construction_system_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )
        
        user_prompt = open(os.path.join(PROMPT_DIR, "emotional_memory_construction_user_prompt_template.txt")).read().format(
            chunk = LifeStory_chunk
        )
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))
        # Generate emotional memory
        with get_openai_callback() as cb:
            ans = self.emotion.invoke(messages)
            self.cost += cb.total_cost
        return ans.content


    def Emotion_analysis(self, query):
        # Analyze the character's current emotion based on the query
        messages = []
        sys_prompt = open(os.path.join(PROMPT_DIR, "generate_personal_emotion_system_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits
        )

        user_prompt = open(os.path.join(PROMPT_DIR, "generate_personal_emotion_user_prompt_template.txt")).read().format(
            query = query
        )
        
        messages.append(SystemMessage(content=sys_prompt))
        messages.append(HumanMessage(content=user_prompt))

        with get_openai_callback() as cb:
            emo_result = self.emotion.invoke(messages)
            self.cost += cb.total_cost
        return emo_result.content
    

class Top_agent:
    # The Top_agent class is responsible for the following functions:
    # (1) Transform questions into descriptive statements and extract key content;
    # (2) Construct and maintain working memory;
    # (3) Answer queries based on working memory.
      
    def __init__(self, character_name, model:str="gpt-4.1-mini", temperature = 1.0, api_base = BASE_URL, api_key = os.environ['OPENAI_API_KEY']):
        self.api_base = api_base
        self.api_key = api_key
        self.cost = 0.0
        self.name = character_name
        self.temperature = temperature
        self.model = model
        flag = False
        with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
        for introduction in introductions:
            if introduction["Name"] == self.name:
                flag = True
                self.infos = introduction['Basic_infos']
                if introduction["Extra"]:
                    # Convert the Extra dictionary to a string format
                    extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
                    character_infos = introduction['Basic_infos'].strip('"\n') 
                    # Append the Extra information to the Basic_infos string and restore the original format
                    character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
                    self.infos = character_infos
                self.personality_traits = introduction['Personality_traits']
                self.biography = introduction['Content']
                break
        if flag == False:
            print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
            sys.exit(1)

        # Initialize three agents
        self.Thinking_Agent = Thinking_agent(character_infos=self.infos, personality_traits=self.personality_traits, character_name=self.name, character_biography=self.biography, model=model, temperature=self.temperature)
        self.Emotion_Agent = Emotion_agent(character_infos=self.infos, personality_traits=self.personality_traits, character_name=self.name, model=model, temperature=self.temperature)
        self.Memory_Agent = Memory_agent(character_name=self.name, model=model, temperature=self.temperature)
        
        self.chat = ChatOpenAI(
            openai_api_key = self.api_key,
            openai_api_base = self.api_base,
            model = model,
            temperature = self.temperature
        )
        system_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            introduction = self.biography,
        )
        ########## for multi-turn conversation ##########
        self.system_prompt = SystemMessage(content=system_prompt)
        self.chat_history = [] # list of list of strings
        self.current_messages = [] # list of list of messages
        #################################################
        
        if not os.path.exists(os.path.join(MEMORY_DIR, self.name)):
            os.mkdir(os.path.join(MEMORY_DIR, self.name))
            print(Fore.RED + "Long memory not found. Constructing long memory..." + Style.RESET_ALL)
            self.long_memory_construction()
            print(Fore.RED + "Long memory construction completed." + Style.RESET_ALL)
    
    def calculate_cost(self):
        total_cost = self.Thinking_Agent.cost + self.Emotion_Agent.cost + self.Memory_Agent.cost + self.cost
        return total_cost
    
    def add_new_attributes(self, new_attributes):
        ## adding new character attributes to the character's profile
        if not isinstance(new_attributes, dict):
            raise ValueError("new attributes must be a dictionary")
        print(Fore.RED + "addng new attributes..." + Style.RESET_ALL)
        with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
            introductions = json.load(file)
            for idx, introduction in enumerate(introductions):
                if introduction["Name"] == self.name:
                    flag = True
                    Extra = introduction["Extra"]
                    Extra.update(new_attributes)
                    introductions[idx]["Extra"] = Extra
                    break
        if flag == False:
            print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
            sys.exit(1)
        with open(INTRODUCTIONS_PATH, "w", encoding="UTF-8") as file:
            json.dump(introductions, file, ensure_ascii=False, indent=4, separators=(',', ': ')) 
        print(Fore.RED + "Task completed." + Style.RESET_ALL) 

    def add_long_memory(self, new_life_story):
        if not isinstance(new_life_story, str):
            raise ValueError("new life story must be a string")
        
        print(Fore.RED + "Constructing long memory..." + Style.RESET_ALL)
        
        index_file_path = os.path.join(MEMORY_DIR, self.name, "index.json")
        with open(index_file_path, 'r') as file:
            index = json.load(file)
        memory_file_path = os.path.join(MEMORY_DIR, self.name, "long_memory.json")
        with open(memory_file_path, 'r') as file:
            long_memory = json.load(file)
            
        memory_content = self.Thinking_Agent.Memory_construction(new_life_story)
        thinking = self.Thinking_Agent.Thinking_Memory_construction(memory_content)
        emotion = self.Emotion_Agent.Memory_construction(new_life_story)
        memory_summary = self.Memory_Agent.Summary(memory_content, emotion)
        memory = {"Memory Content": memory_content, "Thinking": thinking, "Emotion": emotion}
        long_memory_chunk = {"Memory_Summary" : memory_summary , "Memory" : memory}
        keys = [int(k) for k in index.keys()]
        new_key = str(max(keys) + 1).zfill(3)
        index[new_key] = memory_summary
        long_memory[new_key] = long_memory_chunk
        
        self.Memory_Agent.Save_long_memory(long_memory)
        self.Memory_Agent.Save_index_file(index)
        
        print(Fore.RED + "Long memory construction completed. You can start the conversation." + Style.RESET_ALL)
    
    def long_memory_construction(self):
        # Complete process function for constructing long-term memory
        
        # Initialize long_memory and index as dictionaries
        long_memory = {}
        index = {}
        
        # long_memory: {"num1": {"Memory_Summary" : "xxx" , "Memory"：{"Memory Content" : "xxx" , "Thinking" : "xxx" , "Emotion" : "xxx"}},"num2": {"Memory_Summary" : "xxx" , "Memory"：{"Memory Content" : "xxx" , "Thinking" : "xxx" , "Emotion" : "xxx"}}, ... }
        story_path = os.path.join(STORY_DIR, self.name, self.name + ".txt")
        
        # Split life_story into segments
        with open(story_path, 'r', encoding='utf-8') as file:
            num = 0
            
            temp_chunk = file.readline()
            while temp_chunk:
                
                num += 1
                string_num = str(num).zfill(3)
                chunk = temp_chunk
                # Concatenate two paragraphs into one chunk
                temp_chunk = file.readline()
                chunk += temp_chunk
                
                memory_content = self.Thinking_Agent.Memory_construction(chunk)
                thinking = self.Thinking_Agent.Thinking_Memory_construction(memory_content)
                emotion = self.Emotion_Agent.Memory_construction(chunk)
                memory_summary = self.Memory_Agent.Summary(memory_content, emotion)
                memory = {"Memory Content": memory_content, "Thinking": thinking, "Emotion": emotion}
                long_memory_chunk = {"Memory_Summary" : memory_summary , "Memory" : memory}
                index[string_num] = memory_summary
                long_memory[string_num] = long_memory_chunk
                print(Fore.RED + string_num + Style.RESET_ALL, end="\r")  # Display progress
                # print(index)
                # print("--------------------------")
                # print(long_memory)
                # print("--------------------------")
                temp_chunk = file.readline()
        self.Memory_Agent.Save_long_memory(long_memory)
        self.Memory_Agent.Save_index_file(index)
        
    def multi_turn_chat(self):
        # Multi-turn conversation, input "exit" to terminate conversation
        
        System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            introduction = self.biography,
        )
        current_messages = [
            SystemMessage(content=System_prompt)
        ]
        chat_history = []
        while True:
            # Get user input
            query = input()
            if query.lower() == "exit":
                break  
            
            if chat_history:
                history = "\n".join(chat_history)
                context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
                current_messages.append(SystemMessage(content=context))
            else:
                current_messages.append(SystemMessage(content="You're chatting with someone in a coffee shop."))
            
            current_messages.append(HumanMessage(content=query))
            
            memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
            thinking = self.Thinking_Agent.Thinking_analysis(query)
            emotion = self.Emotion_Agent.Emotion_analysis(query)
            # self.cost += self.Memory_Agent.cost + self.Thinking_Agent.cost + self.Emotion_Agent.cost 

            
            if memory_retrieval:
                memory = str(memory_retrieval)
                memory_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_memory_prompt_template.txt")).read().format(
                    memory = memory,
                )
                user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                    thinking = thinking,
                    emotion = emotion,
                    personality_traits = self.personality_traits
                )
                current_messages.append(SystemMessage(content=memory_prompt))
            else:
                # No related content in memory, retrieval failed
                user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                    thinking = thinking,
                    emotion = emotion,
                    personality_traits = self.personality_traits
                )
            
            current_messages.append(SystemMessage(content=user_prompt))
            with get_openai_callback() as cb:
                agents_ans = self.chat(current_messages).content
                # self.cost += cb.total_cost
            print(Fore.GREEN + agents_ans + Style.RESET_ALL)
            chat_history.append("The other person: " + query)
            chat_history.append("You: " + agents_ans)
            
        print(Fore.RED + "The conversation is over." + Style.RESET_ALL)
    
    
    def send_message(self, message:str): # single chat for multi-turn conversation
        # Chat with the character, input a string message
        """
        message: str
        """
        temp_chat_history = []
        temp_current_messages = []

        if self.chat_history:
            for hist in self.chat_history: # e.g., chat_history = [["the other person: hi", "you: hello"], ["the other person: how are you?", "you: I'm fine"]]
                history = "\n".join(hist_string for hist_string in hist)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            temp_current_messages.append(SystemMessage(content=context))
        else:
            temp_current_messages.append(SystemMessage(content="You're chatting with someone in a coffee shop."))

        temp_current_messages.append(HumanMessage(content=message))
        memory_retrieval = self.Memory_Agent.Memory_Retrieval(message)
        thinking = self.Thinking_Agent.Thinking_analysis(message)
        emotion = self.Emotion_Agent.Emotion_analysis(message)
        
        if memory_retrieval:
            memory = str(memory_retrieval)
            memory_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_memory_prompt_template.txt")).read().format(
                memory = memory,
            )
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
            temp_current_messages.append(SystemMessage(content=memory_prompt))
        else:
            # No related content in memory, retrieval failed
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        temp_current_messages.append(SystemMessage(content=user_prompt))
        
        self.current_messages.append(temp_current_messages)
        flattened_messages = [item for sublist in self.current_messages for item in sublist]
        
        # Count tokens in the messages
        enc = encoding_for_model(self.model)
        token_count = 0
        
        # Calculate total token count
        for msg in flattened_messages:
            token_count += len(enc.encode(msg.content))
            
        # Define a maximum token limit (adjust based on your model's context window)
        max_tokens = get_context_limit(self.model)
        
        # Remove oldest messages if token count exceeds limit
        while token_count > max_tokens and len(self.current_messages) > 1:
            # Remove the oldest message batch
            removed_messages = self.current_messages.pop(0)
            removed_history = self.chat_history.pop(0) if self.chat_history else []
            # Subtract removed tokens from count
            for msg in removed_messages:
                token_count -= len(enc.encode(msg.content))
            
                print(f"Removed oldest message batch to stay within context limit. Current token count: {token_count}")
            
        # Rebuild flattened_messages after potential removals
        flattened_messages = [item for sublist in self.current_messages for item in sublist]
        # Flatten the list of lists
        
        with get_openai_callback() as cb:
            agents_ans = self.chat.invoke([self.system_prompt] + flattened_messages).content # solely for the top agent
            self.cost += cb.total_cost
        
        temp_chat_history.append("The other person: " + message)
        temp_chat_history.append("You: " + agents_ans)
        
        self.chat_history.append(temp_chat_history)
        logging.info(f"mem: {self.Memory_Agent.cost}, think: {self.Thinking_Agent.cost}, emo: {self.Emotion_Agent.cost}, top: {self.cost}")
        return agents_ans
        
    def bandwagon_chat(self, query, chat_history=None):
        """
        chat for bandwagon effect
        query: string
        chat_history: [AIMessage, HumanMessage]
        """
        
        System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = self.name,
            basic_information = self.infos,
            personality_traits = self.personality_traits,
            introduction = self.biography,
        )
        messages = [
            SystemMessage(content=System_prompt)
        ]
        if chat_history:
            messages += chat_history
        messages.append(HumanMessage(content=query))
        memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
        thinking = self.Thinking_Agent.Thinking_analysis(query)
        emotion = self.Emotion_Agent.Emotion_analysis(query)
        if memory_retrieval:
            memory = str(memory_retrieval)
            memory_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_memory_prompt_template.txt")).read().format(
                memory = memory
            )
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
            messages.append(SystemMessage(content=memory_prompt))
        else:
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_chat_prompt_template.txt")).read().format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        
        messages.append(SystemMessage(content=user_prompt))
        with get_openai_callback() as cb:
            agents_ans = self.chat.invoke(messages)
            self.cost += cb.total_cost
        return agents_ans.content
    
    def evaluation_chat(self, query):
        # For evaluation only
        
        memory_retrieval = self.Memory_Agent.Memory_Retrieval(query)
        thinking = self.Thinking_Agent.Thinking_analysis(query)
        emotion = self.Emotion_Agent.Emotion_analysis(query)
        if memory_retrieval:
            memory = str(memory_retrieval)
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_evaluation_prompt_template.txt")).read().format(
                memory = memory,
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        else:
            user_prompt = open(os.path.join(PROMPT_DIR, "multi_agent_cognitive_system_simple_evaluation_prompt_template.txt")).read().format(
                thinking = thinking,
                emotion = emotion,
                personality_traits = self.personality_traits
            )
        return user_prompt

def Bandwagon_chat_with_naive_prompt(character_name, query, model:str="gpt-4.1-mini-2025-04-14", chat_history=None):
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    flag = False
    with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            if introduction["Extra"]:
                # Convert the Extra dictionary to a string format
                extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
                character_infos = character_infos.strip('"\n') 
                # Append the Extra information to the Basic_infos string and restore the original format
                character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
            break
    if flag == False:
        print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
        sys.exit(1)
        
    System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    messages = [
        SystemMessage(content=System_prompt)
    ]
    if chat_history:
        messages += chat_history
    messages.append(HumanMessage(content=query))
    agents_ans = chat(messages)
    return agents_ans.content

def Bandwagon_chat_with_blank_model(query, model:str="gpt-4.1-mini-2025-04-14", chat_history=None):
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    messages = []
    if chat_history:
        messages += chat_history
    messages.append(HumanMessage(content=query))
    agents_ans = chat(messages)
    return agents_ans.content

def Bandwagon_chat_with_naive_rag(character_name, query, model:str="gpt-4.1-mini-2025-04-14", chat_history=None):
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    flag = False
    with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            if introduction["Extra"]:
                # Convert the Extra dictionary to a string format
                extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
                character_infos = character_infos.strip('"\n') 
                # Append the Extra information to the Basic_infos string and restore the original format
                character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
            break
    if flag == False:
        print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
        sys.exit(1)
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
        
    loader = PyPDFLoader(os.path.join(STORY_DIR, character_name, character_name + ".pdf"))
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
    )
    docs = text_splitter.split_documents(pages)
    embed_model = OpenAIEmbeddings(
        openai_api_base = BASE_URL, openai_api_key = os.environ['OPENAI_API_KEY']
    )
        
    vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="openai_embed")
        
    System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    messages = [
        SystemMessage(content=System_prompt)
    ]
    if chat_history:
        messages += chat_history
    search_results = vectorstore.similarity_search(query, k=3)
    source_knowledge = "\n".join([x.page_content for x in search_results])
    rag_prompt = open(os.path.join(PROMPT_DIR, "naive_rag_simulacra_prompt_template.txt")).read().format(
            source_knowledge = source_knowledge
        )
    messages.append(SystemMessage(content=rag_prompt))
    messages.append(HumanMessage(content=query))
    agents_ans = chat(messages)
    return agents_ans.content
    

def Multi_turn_chat_with_naive_prompt(character_name, model:str="gpt-4.1-mini-2025-04-14"):
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    flag = False
    with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            if introduction["Extra"]:
                # Convert the Extra dictionary to a string format
                extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
                character_infos = character_infos.strip('"\n') 
                # Append the Extra information to the Basic_infos string and restore the original format
                character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
            break
    if flag == False:
        print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
        sys.exit(1)
        
    System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    current_messages = [
        SystemMessage(content=System_prompt)
    ]
    chat_history = []
    while True:
        query = input()
        if query.lower() == "exit":
            break     
        if chat_history:
            history = "\n".join(chat_history)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            current_messages.append(SystemMessage(content=context))
        else:
            current_messages.append(SystemMessage(content="You're chatting with someone in a coffee shop."))

        current_messages.append(HumanMessage(content=query))
        agents_ans = chat(current_messages).content
        print(Fore.GREEN + agents_ans + Style.RESET_ALL)
        chat_history.append(query)
        chat_history.append(agents_ans)
            
    print(Fore.RED + "The conversation is over." + Style.RESET_ALL)

def Multi_turn_chat_with_blank_model(model:str="gpt-4.1-mini-2025-04-14"):
    # blank model, which does not know anything about the character.
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    current_messages = []
    chat_history = []
    while True:
        query = input()
        if query.lower() == "exit":
            break     
        if chat_history:
            history = "\n".join(chat_history)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            current_messages.append(SystemMessage(content=context))
        else:
            current_messages.append(SystemMessage(content="You're chatting with someone in a coffee shop."))

        current_messages.append(HumanMessage(content=query))
        agents_ans = chat(current_messages).content
        print(Fore.GREEN + agents_ans + Style.RESET_ALL)
        chat_history.append(query)
        chat_history.append(agents_ans)
            
    print(Fore.RED + "The conversation is over." + Style.RESET_ALL)

def Multi_turn_chat_with_naive_rag(character_name, model:str="gpt-4.1-mini-2025-04-14"):
    chat = ChatOpenAI(
        openai_api_key = os.environ['OPENAI_API_KEY'],
        openai_api_base = BASE_URL,
        model = model
    )
    flag = False
    with open(INTRODUCTIONS_PATH, "r", encoding="UTF-8") as file:
        introductions = json.load(file)
    for introduction in introductions:
        if introduction["Name"] == character_name:
            flag = True
            character_infos = introduction['Basic_infos']
            if introduction["Extra"]:
                # Convert the Extra dictionary to a string format
                extra_info_str = ', '.join([f"{key}: {value}" for key, value in introduction["Extra"].items()])
                character_infos = character_infos.strip('"\n') 
                # Append the Extra information to the Basic_infos string and restore the original format
                character_infos = f"\"\"\n" + character_infos.rstrip(".") + f", {extra_info_str}.\n\"\"\n"
            personality_traits = introduction['Personality_traits']
            character_biography = introduction['Content']
            break
    if flag == False:
        print(Fore.RED + "Can not find the information." + Style.RESET_ALL)
        sys.exit(1)
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
        
    loader = PyPDFLoader(os.path.join(STORY_DIR, character_name, character_name + ".pdf"))
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
    )
    docs = text_splitter.split_documents(pages)
    embed_model = OpenAIEmbeddings(
        openai_api_base = BASE_URL, openai_api_key = os.environ['OPENAI_API_KEY']
    )
        
    vectorstore = Chroma.from_documents(documents=docs, embedding=embed_model, collection_name="openai_embed")
        
    System_prompt = open(os.path.join(PROMPT_DIR, "naive_simulacra_prompt_template.txt")).read().format(
            character_name = character_name,
            basic_information = character_infos,
            personality_traits = personality_traits,
            introduction = character_biography,
        )
    current_messages = [
        SystemMessage(content=System_prompt)
    ]
    chat_history = []
    while True:
        query = input()

        if query.lower() == "exit":
            break  
        search_results = vectorstore.similarity_search(query, k=3)
        source_knowledge = "\n".join([x.page_content for x in search_results])
        rag_prompt = open(os.path.join(PROMPT_DIR, "naive_rag_simulacra_prompt_template.txt")).read().format(
                        source_knowledge = source_knowledge
        )
        current_messages.append(SystemMessage(content=rag_prompt))
        
        if chat_history:
            history = "\n".join(chat_history)
            context = "You're chatting with someone in a coffee shop. This is your conversation record: <<<\n" + history + ">>>"
            current_messages.append(SystemMessage(content=context))
        else:
            current_messages.append(SystemMessage(content="You're chatting with someone in a coffee shop."))
            
        user_prompt = f"The one you are chatting with said:<<<{query}>>>"
        current_messages.append(HumanMessage(content=user_prompt))
        agents_ans = chat(current_messages).content
        print(Fore.GREEN + agents_ans + Style.RESET_ALL)
        chat_history.append(query)
        chat_history.append(agents_ans)
            
    print(Fore.RED + "The conversation is over." + Style.RESET_ALL)
  
def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--character_name", type=str, required=True, help="Name of the character")
    parser.add_argument("--method", type=str, choices=["prompt", "rag", "macm", "none"], required=True, help="Method of conversation: prompt, rag, none or macm")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")

    # Parse arguments
    args = parser.parse_args()

    # Check if character_name is in CHARACTERS
    if args.character_name not in CHARACTERS:
        print(f"Error: {args.character_name} is not in the CHARACTERS.")
        sys.exit(1)

    # Execute based on method
    if args.method == "rag":
        print(Fore.RED + f"Starting multi-turn chat with RAG method for {args.character_name}. Type 'exit' to end the chat." + Style.RESET_ALL)
        Multi_turn_chat_with_naive_rag(args.character_name)
    elif args.method == "macm":
        print(Fore.RED + f"Starting multi-turn chat with MACM method for {args.character_name}. Type 'exit' to end the chat." + Style.RESET_ALL)
        agent = Top_agent(args.character_name, temperature=args.temperature)
        agent.multi_turn_chat()
    elif args.method == "prompt":
        print(Fore.RED + f"Starting multi-turn chat with prompt method for {args.character_name}. Type 'exit' to end the chat." + Style.RESET_ALL)
        Multi_turn_chat_with_naive_prompt(args.character_name)
    else:
        print(Fore.RED + f"Starting multi-turn chat with blank model, which does not know anything about the character. Type 'exit' to end the chat." + Style.RESET_ALL)
        Multi_turn_chat_with_blank_model()
        
    # agent = Top_agent("Mary Jones")
    # agent.add_new_attributes({"Favorite Color": "Green", "Favorite Author": "Ada Lovelace"})
    # agent.add_long_memory("Mary's favorite color is green, and her favorite author is Ada Lovelace.")
    

if __name__ == "__main__":
    main()
