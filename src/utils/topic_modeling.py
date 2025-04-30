'''Zero-shot topic modeling Modeler class.'''

from dataclasses import dataclass, field
from typing import List, Optional, TypeAlias
from top2vec import Top2Vec
from langchain.schema import Document
from langchain_text_splitters import TokenTextSplitter
from tqdm import tqdm
from transformers import AutoTokenizer, PegasusTokenizer
from utils import setup_logger

AbstractTokenizer: TypeAlias = AutoTokenizer | PegasusTokenizer

logger = setup_logger(__name__)


@dataclass
class TopicModeler:
    '''
    A wrapper around Top2Vec that handles preprocessing, chunking,
    and training on a list of input texts.

    Attributes:
        texts: Raw input documents to be processed.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Overlap between chunks.
        speed: Top2Vec training speed ('learn', 'deep-learn', 'fast-learn').
        workers: Number of threads to use for training.
    '''

    texts: List[str]
    tokenizer: AbstractTokenizer = None
    _chunk_size: int = 1024
    _chunk_overlap: int = 64
    speed: str = 'learn'
    workers: int = 4
    model: Top2Vec = field(init=False)
    processed_texts: List[str] = field(init=False)
    hierarchy: Optional[List[int]] = field(default=None, init=False)

    def __post_init__(self):
        '''
        Automatically called after initialization. Triggers preprocessing and model training.
        '''
        if self.tokenizer:
            logger.info('Tokenizer injected with max length: %s ', self.tokenizer.model_max_length)
            self._chunk_size = min(1024, self.tokenizer.model_max_length)
        logger.info('Initializing TopicModeler and starting preprocessing...')
        self._preprocess()
        logger.info('Preprocessing complete. Starting model training...')
        self._train_model()
        logger.info('Model training complete.')

    def _preprocess(self):
        '''
        Splits input texts into smaller overlapping chunks using LangChain's TokenTextSplitter.
        This ensures chunks respect token boundaries for language models.
        '''
        logger.info(
            'Chunking input documents with token size %d and overlap %d',
            self._chunk_size,
            self._chunk_overlap,
        )
        docs = [
            Document(page_content=text) for text in tqdm(self.texts, desc='Preparing documents')
        ]
        splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=self.tokenizer, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        self.processed_texts = [
            chunk.page_content for chunk in tqdm(chunks, desc='Splitting documents')
        ]
        logger.info(
            'Generated %d total chunks from %d original texts',
            len(self.processed_texts),
            len(self.texts),
        )

    def _train_model(self):
        '''
        Trains the Top2Vec model on the preprocessed chunks.
        '''
        self.model = Top2Vec(self.processed_texts, speed=self.speed, workers=self.workers)

    def get_topics(self, num_topics: int = None) -> List[dict]:
        """
        Returns the top keywords from the discovered topics.

        Args:
            num_topics: Number of top topics to return. If None, returns all topics.

        Returns:
            A list of dictionaries, each containing a topic number and its keywords.
        """
        topic_count = self.model.get_num_topics() if num_topics is None else num_topics
        topic_words, _, topic_nums = self.model.get_topics(topic_count)
        topics = []

        for topic_num, words in zip(topic_nums, topic_words):
            topics.append({"topic_id": topic_num, "keywords": words})

        return topics

    def show_wordcloud(self, topic_id: int = 0):
        '''
        Displays a word cloud for a given topic ID.

        Args:
            topic_id: The topic index to visualize.
        '''
        self.model.generate_topic_wordcloud(topic_id)

    def reduce_topics(self, num_topics: int = 1):
        '''
        Reduces the number of topics using hierarchical topic reduction and returns the reduced topics' keywords.

        Args:
            num_topics: The number of final reduced topics.

        Returns:
            A list of dictionaries containing the reduced topics and their keywords.
        '''
        logger.info('Reducing topics to %d using hierarchical topic reduction...', num_topics)
        self.model.hierarchical_topic_reduction(num_topics=num_topics)
        self.hierarchy = self.model.get_topic_hierarchy()

        reduced_topics = []
        for reduced_topic in range(num_topics):
            original_topics = [i for i, x in enumerate(self.hierarchy) if x == reduced_topic]
            topic_keywords = []

            for original_topic in original_topics:
                topic_words, _, _ = self.model.get_topics(1, topic_num=original_topic)
                topic_keywords.extend(topic_words[0])

            topic_keywords = list(set(topic_keywords))

            reduced_topics.append({"reduced_topic_id": reduced_topic, "keywords": topic_keywords})

        return reduced_topics
