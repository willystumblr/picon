from litellm.cost_calculator import completion_cost
from picon.utils import get_completion
from picon.schemas import IntervieweeResponse
from picon.env.interviewee_simulator.base_interviewee_simulator import BaseIntervieweeSimulator
import logging


class GenericAgentSimulator(BaseIntervieweeSimulator):
    """
    Universal interviewee simulator that works with any OpenAI-compatible API endpoint.

    All evaluation targets — whether cloud LLM APIs, self-hosted vLLM, or custom
    wrapping servers — are accessed through the same /v1/chat/completions interface.

    Two modes:
        - External agent: provide api_base (model is optional, defaults to placeholder)
        - LLM persona: provide model (and optionally persona, api_key)

    At least one of model or api_base must be provided.

    Optional kwargs:
        - model (str): Model name. Required when api_base is not set.
        - persona (str): System prompt. Can be empty if the server manages persona internally.
        - api_base (str): API endpoint URL. None = litellm default routing.
        - api_key (str): API key. Defaults to "no-key".
        - name (str): Interviewee name. Defaults to "GenericAgent".
        - user_message_template (str): Format string for user messages.
            Defaults to "{message}" (no wrapping).
            Examples: "user request: {message}", "### QUESTION ###\\n{message}\\n\\n### YOUR RESPONSE ###"
        - completion_kwargs (dict): Extra kwargs for get_completion (temperature, top_p, max_tokens, etc.)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs.get('name', 'GenericAgent')
        self.user_message_template = kwargs.get('user_message_template', '{message}')
        self.extra_completion_kwargs = kwargs.get('completion_kwargs', {})

        # Build api_base: explicit > host+port > None (litellm default)
        self.api_base = kwargs.get('api_base')
        if not self.api_base:
            host = kwargs.get('simulator_host')
            port = kwargs.get('port')
            if host and port:
                self.api_base = f"http://{host}:{port}/v1"

        self.api_key = kwargs.get('api_key', 'no-key')

        # Resolve model name:
        # - If api_base is set, routing model always uses "openai/" prefix so litellm
        #   sends an OpenAI-format request to the custom endpoint.
        # - Original model kwarg is preserved separately for token-limit lookup.
        # - If no api_base, model is required (litellm needs it for routing)
        _orig_model = kwargs.get('model')
        model = kwargs.get('model')
        if self.api_base:
            if not model:
                model = "openai/default"
            elif model.startswith("openai/"):
                pass  # already correct
            elif '/' in model:
                # e.g. "gemini/gemini-3-flash-preview" → "openai/gemini-3-flash-preview"
                model = f"openai/{model.split('/', 1)[1]}"
            else:
                model = f"openai/{model}"
        else:
            assert model, "GenericAgentSimulator requires 'model' when 'api_base' is not provided"
        self.client_or_model = model

        # Initialize chat history
        persona = kwargs.get('persona', '')
        self.history = [{"role": "system", "content": persona}] if persona else []

        # Determine max tokens: try original model name first (e.g. "gemini/..."),
        # then routing model, then fall back to 8192.
        # Also sets self.tokenizer (required by base _count_tokens).
        from litellm import get_model_info, token_counter
        self.tokenizer = token_counter
        self.max_tokens = 8192
        for _m in filter(None, [_orig_model, self.client_or_model]):
            try:
                self.max_tokens = get_model_info(_m)['max_input_tokens']
                break
            except Exception:
                continue
        else:
            logging.warning(f"Could not determine max tokens for {self.client_or_model}. Using default 8192.")

        # For custom endpoints: get accurate max_model_len and load actual tokenizer
        self._endpoint_tokenizer = None
        if self.api_base:
            try:
                import requests
                resp = requests.get(f"{self.api_base}/models", timeout=5)
                model_data = resp.json()["data"][0]
                if "max_model_len" in model_data:
                    self.max_tokens = model_data["max_model_len"]
                model_id = model_data.get("id") or model_data.get("root")
                if model_id:
                    from transformers import AutoTokenizer
                    self._endpoint_tokenizer = AutoTokenizer.from_pretrained(model_id)
                    logging.info(f"Loaded tokenizer for {model_id}, max_tokens={self.max_tokens}")
            except Exception as e:
                logging.warning(f"Could not query endpoint for model info: {e}")

    def _count_tokens(self, messages):
        if self._endpoint_tokenizer is not None:
            input_ids = self._endpoint_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return len(input_ids)
        count = super()._count_tokens(messages)
        # litellm token_counter returns unreliable results for unknown custom model IDs.
        # Use character-based estimate (÷3) as a floor to prevent underestimation.
        char_estimate = sum(len(str(m.get("content", ""))) for m in messages) // 3
        return max(count, char_estimate)


    def _get_response(self, message: str) -> IntervieweeResponse:
        formatted = self.user_message_template.format(message=message)
        self.history.append({"role": "user", "content": formatted})
        self._truncate_history()

        call_kwargs = {
            "model": self.client_or_model,
            "messages": self.history,
        }
        if self.api_base:
            call_kwargs["api_base"] = self.api_base
        if self.api_key and self.api_key != "no-key":
            call_kwargs["api_key"] = self.api_key
        call_kwargs.update(self.extra_completion_kwargs)

        # Dynamically cap max_tokens so input + output never exceeds model limit
        if "max_tokens" not in call_kwargs:
            if self._endpoint_tokenizer is not None:
                try:
                    input_tokens = self._count_tokens(self.history)
                    available = self.max_tokens - input_tokens - 128
                    call_kwargs["max_tokens"] = max(64, available)
                except Exception:
                    call_kwargs["max_tokens"] = 1024
            elif self.api_base:
                # Custom endpoint without a loaded tokenizer: token counting is unreliable.
                # Cap conservatively — interview responses never need > 1024 tokens.
                call_kwargs["max_tokens"] = 1024
            else:
                try:
                    input_tokens = self._count_tokens(self.history)
                    available = self.max_tokens - input_tokens - 128
                    call_kwargs["max_tokens"] = max(64, available)
                except Exception:
                    pass

        # Hard cap: for custom endpoints token counting is unreliable, so cap output firmly.
        if self.api_base:
            call_kwargs["max_tokens"] = min(call_kwargs.get("max_tokens", 1024), 1024)

        # On ContextWindowExceededError, parse actual input_tokens from error message and retry once.
        import re as _re
        for _attempt in range(2):
            try:
                res = get_completion(**call_kwargs)
                break
            except Exception as _e:
                m = _re.search(r'has (\d+) input tokens', str(_e))
                if m and _attempt == 0:
                    actual_input = int(m.group(1))
                    safe_max = max(64, self.max_tokens - actual_input - 64)
                    call_kwargs["max_tokens"] = safe_max
                    # Also trim history to fit within model limit
                    while len(self.history) > 2 and actual_input + safe_max > self.max_tokens:
                        self.history.pop(1)
                        if len(self.history) > 1:
                            self.history.pop(1)
                        actual_input = max(64, actual_input - 200)
                    call_kwargs["messages"] = self.history
                else:
                    raise

        try:
            self.cost += completion_cost(completion_response=res)
        except Exception:
            pass

        response = res.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": response})

        self._ai_check(message, response)

        return IntervieweeResponse(question=message, content=response)
