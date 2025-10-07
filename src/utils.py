import json
import os
from typing import Any
import logging
import time
import sys
from litellm import completion

def read_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)
def write_json(data: Any, file_path: str) -> None:
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
            assert response is not None and response.choices, f"Invalid response from completion API : {response}"
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