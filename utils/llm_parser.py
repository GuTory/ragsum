import re
import json
import logging
import requests
from dataclasses import dataclass

@dataclass
class DeepSeekResponse:
    thinking: str
    response: str

class DeepSeekAPI:
    def __init__(
        self,
        model: str = 'deepseek-r1:8b',
        url: str = 'http://localhost:11434/api/generate',
        stream: bool = False
    ):
        self.url = url
        self.model = model
        self.url = url
        self.stream = stream
        self.response: DeepSeekResponse = None

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                self.url, 
                json= {
                    'model': self.model,
                    'prompt': prompt,
                    'stream': self.stream
                }
            )
            response.raise_for_status()
            data = response.json()

            if 'response' in data:
                logging.info(f'Response received from model {self.model}')
                self.process_text(data['response'])
                return self.DeepSeekResponse.response
            else:
                logging.warning(f'No response field in API response: {data}')
                return 'No response received.'
    
        except requests.exceptions.RequestException as e:
            request_error = f'HTTP request failed: {e}'
            logging.error(request_error)
            return request_error
        except ValueError as e:
            error_message = f'JSON decoding failed: {e}'
            logging.error(error_message)
            return error_message

    def process_text(self, response: str):
        """
        Extracts the thinking phase between <think> tags and the final response after </think>.
        """
        match = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
        if match:
            self.DeepSeekResponse = DeepSeekResponse(
                thinking=match.group(1).strip(),
                response=match.group(2).strip()
            )
        else:
            logging.error('no match in thinking regex')
            self.DeepSeekResponse = None