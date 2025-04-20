'''
Module for interacting with locally running DeepSeek model
'''

import re
import logging
from dataclasses import dataclass
from typing import Dict, Any
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


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
        model_name: str = 'deepseek-r1:8b',
        url: str = 'http://localhost:11434/api/generate',
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.url = url
        self.model_name = model_name
        self.stream = False
        self.response: DeepSeekResponse = None
        self.timeout = timeout
        self.max_retries = max_retries

    def __str__(self):
        return (
            f'DeepSeekAPI(model_name={self.model_name}, url={self.url}, '
            f'stream={self.stream}, timeout={self.timeout}, max_retries={self.max_retries})'
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def generate(self, prompt: str) -> str:
        '''
        Asynchronous function that calls the locally running model through HTTP.

        Args:
            prompt: The input prompt to send to the model

        Returns:
            Generated text response from the model

        Raises:
            aiohttp.ClientError: For HTTP request failures
            ValueError: For JSON decoding issues
            TimeoutError: When the request times out
        '''
        try:
            async with aiohttp.ClientSession() as session:
                request_data = {'model': self.model_name, 'prompt': prompt, 'stream': self.stream}
                logging.debug('Sending request to %s with model %s', self.url, self.model_name)
                return await self._handle_non_streaming(session, request_data)

        except aiohttp.ClientError as e:
            request_error = f'HTTP request failed: {e}'
            logging.error(request_error)
            raise
        except ValueError as e:
            error_message = f'JSON decoding failed: {e}'
            logging.error(error_message)
            raise

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

    async def _handle_non_streaming(
        self, session: aiohttp.ClientSession, request_data: Dict[str, Any]
    ) -> str:
        '''Handle non-streaming response format.'''
        async with session.post(self.url, json=request_data, timeout=self.timeout) as response:
            response.raise_for_status()
            data = await response.json()

        if 'response' in data:
            logging.info('Response received from model %s', self.model_name)
            self.process_text(data['response'])
            return self.response.response

        logging.warning('No response field in API response: %s', data)
        return None
