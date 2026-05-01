import json
import logging
from transformers import AutoTokenizer
from picon.utils import get_completion
from picon.schemas import IntervieweeResponse
from picon.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator


class ConsistentLLMSimulator(BaseIntervieweeSimulator):
    """
    Replicates ConsistentLLMSimulator from prev_code.

    Persona file (JSON) must contain:
        persona, name, counterpart_name, instruction, model_path
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        persona_arg = kwargs.get("persona", "")
        if persona_arg and persona_arg.strip().startswith("{"):
            data = json.loads(persona_arg)
        else:
            raise ValueError("ConsistentLLMSimulator requires persona to be a JSON string.")

        self.persona = data["persona"]
        self.name = data["name"]
        self.counterpart_name = data["counterpart_name"]
        self.instruction = data["instruction"]
        model_path = data["model_path"]

        self.host = kwargs.get("simulator_host", "localhost")
        self.port = kwargs["api_base"].rstrip("/").rsplit(":", 1)[-1].split("/")[0] \
            if "api_base" in kwargs else str(kwargs["port"])
        self.api_base = kwargs.get("api_base", f"http://{self.host}:{self.port}/v1")
        self.simulator_model = kwargs.get("model") or "openai/default"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.client_or_model = model_path  # used by base for get_max_token fallback
        try:
            import requests
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            self.max_tokens = resp.json()["data"][0].get("max_model_len", 8192)
        except Exception:
            self.max_tokens = 8192
        self.history = []
        self.prompt_flag = "Your conversation so far is below:\nConversation: \n"

    def _get_response(self, message: str) -> IntervieweeResponse:
        self.history.append(f"Interviewer: {message}")

        while True:
            user_content = self.prompt_flag + "\n".join(self.history) + self.instruction
            messages = [
                {"role": "system", "content": self.persona},
                {"role": "user",   "content": user_content},
            ]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if len(input_ids) + 1024 <= self.max_tokens:
                break
            if len(self.history) <= 2:
                break
            self.history = self.history[2:]

        input_len = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        available = self.max_tokens - input_len - 64
        output_tokens = max(64, min(1024, available))

        res = get_completion(
            model=self.simulator_model,
            messages=messages,
            api_base=self.api_base,
            max_tokens=output_tokens,
        )
        response = res.choices[0].message.content.strip()
        self.history.append(f"{self.name}: {response}")

        self._ai_check(message, response)
        return IntervieweeResponse(question=message, content=response)
