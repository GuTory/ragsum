'''
Module for interacting with locally running DeepSeek model
'''

import re
import logging
from dataclasses import dataclass
import requests


@dataclass
class DeepSeekResponse:
    '''
    DeepSeek response data holder: thinking and final response string
    '''
    thinking: str
    response: str


class DeepSeekAPI:
    '''
    API class for locally running DeepSeek model
    '''

    def __init__(
        self,
        model: str = 'deepseek-r1:8b',
        url: str = 'http://localhost:11434/api/generate',
        stream: bool = False,
    ):
        self.url = url
        self.model = model
        self.url = url
        self.stream = stream
        self.response: DeepSeekResponse = None

    def generate(self, prompt: str) -> str:
        '''
        Generator function that calls the locally running model through HTTP
        '''
        try:
            response = requests.post(
                self.url,
                json={'model': self.model, 'prompt': prompt, 'stream': self.stream},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if 'response' in data:
                logging.info('Response received from model %s', self.model)
                self.process_text(data['response'])
                return self.response.response
            logging.warning('No response field in API response: %s', data)
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
        '''
        Extracts the thinking phase between <think> tags and the final response after </think>.
        '''
        match = re.search(r'<think>(.*?)</think>(.*)', response, re.DOTALL)
        if match:
            self.response = DeepSeekResponse(
                thinking=match.group(1).strip(), response=match.group(2).strip()
            )
        else:
            logging.error('no match in thinking regex')
            self.response = None
