from PyCharacterAI import get_client
from PyCharacterAI.exceptions import SessionClosedError
import asyncio
import threading
from src.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
from src.utils import get_completion
from src.schemas import Action, IntervieweeResponse
from litellm.cost_calculator import completion_cost
import logging
import time

CAI_TIMEOUT = 30  # seconds


class CharacterAISimulator(BaseIntervieweeSimulator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert 'character_id' in kwargs, "Character AI requires character_id parameter"
        assert 'user_id' in kwargs, "Character AI requires user_id parameter"

        self.char_id = kwargs['character_id']
        self.user_id = kwargs['user_id']

        # Create a persistent event loop running in a background thread.
        # The PyCharacterAI client binds its aiohttp session to the loop it
        # was created on, so we must reuse the same loop for all operations.
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

        try:
            self._run_on_loop(self._setup_client_and_chat(kwargs['user_id'], kwargs['character_id']))
            assert self.chat_id is not None, "Chat ID must be set after setup"
            logging.info(f"CharacterAI client and chat session established. Chat ID: {self.chat_id}")
        except SessionClosedError as e:
            logging.error(f"Session closed error: {e}")
            self.client_or_model = None
            self.chat_id = None

    def _run_on_loop(self, coro, timeout=CAI_TIMEOUT):
        """Submit a coroutine to the persistent event loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def _get_response(self, message: str) -> IntervieweeResponse:
        time.sleep(0.5)  # 500ms delay

        async def _send():
            return await asyncio.wait_for(
                self.client_or_model.chat.send_message(
                    character_id=self.char_id,
                    chat_id=self.chat_id,
                    text=message
                ),
                timeout=CAI_TIMEOUT
            )

        response = self._run_on_loop(_send(), timeout=CAI_TIMEOUT + 5)
        response = response.get_primary_candidate().text

        self._ai_check(message, response)

        return IntervieweeResponse(
            question=message,
            content=response
        )

    async def _setup_client_and_chat(self, user_id: str, char_id: str ):
        try:
            self.client_or_model = await asyncio.wait_for(
                get_client(user_id), timeout=CAI_TIMEOUT
            )

            me_task = asyncio.create_task(self.client_or_model.account.fetch_me())
            chat_task = asyncio.create_task(self.client_or_model.chat.create_chat(char_id))

            me, (chat, greeting_message) = await asyncio.wait_for(
                asyncio.gather(me_task, chat_task), timeout=CAI_TIMEOUT
            )

            self.chat_id = chat.chat_id

        except Exception as e:
            logging.error(f"Failed to set up the cai client: {e}")
            await self.client_or_model.close_session()

    async def close(self):
        if hasattr(self, 'client_or_model') and self.client_or_model is not None:
            await self.client_or_model.close_session()
            self.client_or_model = None
            self.chat_id = None

    def shutdown(self):
        """Close the client session and stop the persistent event loop."""
        try:
            if hasattr(self, '_loop') and self._loop.is_running():
                self._run_on_loop(self.close(), timeout=CAI_TIMEOUT)
        except Exception as e:
            logging.warning(f"Error closing CharacterAI session: {e}")
        finally:
            if hasattr(self, '_loop') and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)
            if hasattr(self, '_loop_thread'):
                self._loop_thread.join(timeout=5)

    def calculate_cost(self) -> float:
        # CharacterAI does not provide cost details, so we return 0.0 here.
        return self.cost
