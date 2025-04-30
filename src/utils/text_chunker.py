'''Text Chunker for transcripts'''

from dataclasses import dataclass, field
from typing import List, TypeAlias
from tqdm import tqdm
from langchain_text_splitters import TokenTextSplitter
from transformers import AutoTokenizer, PegasusTokenizer

AutoOrPegasusTokenizer: TypeAlias = AutoTokenizer | PegasusTokenizer

from utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class TextChunker:
    '''
    A utility class for splitting text into chunks using LangChain's TokenTextSplitter.
    Each chunk will have a specified prefix, and the chunk size will be adjusted to account for the prefix length.
    '''

    tokenizer: AutoOrPegasusTokenizer
    prefix: str = ''
    _adjusted_chunk_size: int = field(init=False)
    _chunk_overlap: int = field(init=False)
    _splitter: TokenTextSplitter = field(init=False)

    def __post_init__(self):
        '''
        Post-init function to initialize adjusted chunk size, overlap, and the TokenTextSplitter.
        '''
        if not self.tokenizer:
            logger.error('Tokenizer not provided.')
            raise ValueError('Please provide a tokenizer.')
        if not self.tokenizer.model_max_length:
            chunk_size = 1024
        else:
            chunk_size = min(1024, self.tokenizer.model_max_length)

        self._adjusted_chunk_size = max(1, chunk_size - len(self.prefix.split()))
        self._chunk_overlap = max(1, self._adjusted_chunk_size // 10)

        self._splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer,
            chunk_size=self._adjusted_chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

        logger.info(
            f'Initialized TextChunker with chunk_size={self._adjusted_chunk_size}, '
            f'chunk_overlap={self._chunk_overlap}, prefix="{self.prefix}"'
        )

    def chunk_text(self, text: str) -> List[str]:
        '''
        Splits the input text into chunks, each prefixed with the specified prefix.

        Args:
            text (str): Text to be chunked.

        Returns:
            List[str]: A list of text chunks, each with the specified prefix.
        '''
        logger.info('Starting text chunking...')
        chunks = self._splitter.split_text(text)

        chunked_with_prefix = []
        for chunk in tqdm(chunks, desc="Chunking text"):
            chunked_with_prefix.append(f'{self.prefix}{chunk}')

        logger.info(f'Text successfully split into {len(chunked_with_prefix)} chunks.')
        return chunked_with_prefix
