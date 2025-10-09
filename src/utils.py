import json
import os
from typing import List, Any
import logging
import time
import sys
from litellm import completion
from google import genai

def read_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)
def write_json(data: Any, file_path: str) -> None:
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_completion(model: str, messages: list, temperature: float = 0.0, max_retries=3, **kwargs):
    for attempt in range(1, max_retries+1):
        try:
            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            assert response.choices and len(response.choices) > 0, f"Invalid response from completion API: No choices : {response}"
            return response
        except Exception as e:
            logging.error(f"Error during completion: {e}")
            wait_time = 30
            logging.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time) 
    logging.exception("Max retries reached. Raising exception.")
    raise RuntimeError("Failed to get completion after multiple attempts.")


def setup_logging(log_to_file: bool, process_name: str = None):
    if log_to_file:
        os.makedirs(f'logs/{time.strftime("%Y-%m-%d")}', exist_ok=True)
        log_filename = f'logs/{time.strftime("%Y-%m-%d")}/{process_name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(levelname)s  %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ])
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s  %(levelname)s  %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    for noisy in ("LiteLLM", "httpx", "google", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)    
        
def _batch_preprocess(messages_lists: List, provider: str="google-genai"):
    # Implement your batch preprocessing logic here
    """format messages for batch processing
    inline_requests = [
        {
            'contents': [{
                'parts': [{'text': 'Tell me a one-sentence joke.'}],
                'role': 'user'
            }]
        },
        {
            'contents': [{
                'parts': [{'text': 'Why is the sky blue?'}],
                'role': 'user'
            }]
        }
    ]
    """
    if provider == "google-genai":
        request_list = []
        for messages in messages_lists:
            contents = []
            system_instruction = {}
            for message in messages:
                if message['role'] == 'system':
                    system_instruction['parts'] = [{'text': message['content']}]
                else:
                    contents.append({
                        'parts': [{'text': message['content']}],
                        'role': message['role']
                    })
            request_list.append({
                'contents': contents,
                'config': {
                    'system_instruction': system_instruction,
                    "thinking_config": {"thinking_budget":1024}
                }
            })
        return request_list
    else:
        ## TODO
        return None

def batch_request_gemini(display_name:str, messages: List=None, model: str="gemini-2.5-flash", inline_request: bool=True, file_request: bool=False, input_files: str=None, sub_batch: bool=False):
    if inline_request:
        # Implement inline request logic here
        assert messages is not None, "Messages must be provided for inline requests."
        results = []
        if sub_batch:
            batch_size = 500
        else:
            batch_size = len(messages)
        
        for i in range(0, len(messages), batch_size):
            provider = "google-genai" if 'gemini' in model else "other-provider" # placeholder
            request_list = _batch_preprocess(messages[i:i + batch_size], provider=provider)
            completion_model = f"gemini/{model}" if "gemini" in model else model

            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            batch_job = client.batches.create(
                model=model,
                src=request_list,
                config={
                    'display_name': display_name,
                },
            )
            
            while True:
                job = client.batches.get(name=batch_job.name)  # Get the latest status
                state = job.state.name
                logging.info(f"Current state: {state}")

                if state == "JOB_STATE_SUCCEEDED":
                    logging.info("✅ Job succeeded!")
                    if job.dest and job.dest.inlined_responses:
                        for i, inline_resp in enumerate(job.dest.inlined_responses):
                            if inline_resp.response and inline_resp.response.text:
                                results.append(inline_resp.response.text)    
                    break
                elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
                    raise RuntimeError(f"Batch job failed: {state}")
                else:
                    time.sleep(10)

        return results
    if file_request and input_files:
        # Implement file request logic here
        pass