from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError
import asyncio
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from litellm.cost_calculator import completion_cost
import nest_asyncio
import logging
import time

nest_asyncio.apply()

class CharacterAISimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'character_id' in kwargs, "Character AI requires character_id parameter"
        assert 'user_id' in kwargs, "Character AI requires user_id parameter"
        
        self.char_id = kwargs['character_id']
        self.user_id = kwargs['user_id']
        
        try:
            asyncio.run(self._setup_client_and_chat(kwargs['user_id'], kwargs['character_id']))
            assert self.chat_id is not None, "Chat ID must be set after setup"
            logging.info(f"CharacterAI client and chat session established. Chat ID: {self.chat_id}")
        except SessionClosedError as e:
            logging.error(f"Session closed error: {e}")
            self.client_or_model = None
            self.chat_id = None
    
    def _get_response(self, message: str) -> IntervieweeResponse:
        time.sleep(0.5)  # 500ms delay
           
        response = asyncio.run(
            self.client_or_model.chat.send_message(
                character_id=self.char_id, 
                chat_id=self.chat_id, 
                text=message
            )
        )
        response = response.get_primary_candidate().text
        
        self._ai_check(message, response)
        
        return IntervieweeResponse(
            question=message,
            content=response
        )
    
    async def _setup_client_and_chat(self, user_id: str, char_id: str ):
        try:
            self.client_or_model = await get_client(user_id)
            
            me_task = asyncio.create_task(self.client_or_model.account.fetch_me())
            chat_task = asyncio.create_task(self.client_or_model.chat.create_chat(char_id))
            
            me, (chat, greeting_message) = await asyncio.gather(me_task, chat_task)
            
            self.chat_id = chat.chat_id
            
        except Exception as e:
            logging.error(f"Failed to set up the cai client: {e}")
            await self.client_or_model.close_session()
    
    async def close(self):
        if hasattr(self, 'client_or_model') and self.client_or_model is not None:
            await self.client_or_model.close_session()
            self.client_or_model = None
            self.chat_id = None
        
    def calculate_cost(self) -> float:
        # CharacterAI does not provide cost details, so we return 0.0 here.
        return self.cost
