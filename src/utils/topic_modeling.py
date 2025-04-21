'''
Zero-shot topic modeling Modeler class.
'''

from dataclasses import dataclass, field
from typing import List, Optional
from top2vec import Top2Vec
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from utils import setup_logger

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
    chunk_size: int = 512
    chunk_overlap: int = 64
    speed: str = 'learn'
    workers: int = 4
    model: Top2Vec = field(init=False)
    processed_texts: List[str] = field(init=False)
    hierarchy: Optional[List[int]] = field(default=None, init=False)

    def __post_init__(self):
        '''
        Automatically called after initialization. Triggers preprocessing and model training.
        '''
        logger.info('Initializing TopicModeler and starting preprocessing...')
        self._preprocess()
        logger.info('Preprocessing complete. Starting model training...')
        self._train_model()
        logger.info('Model training complete.')

    def _preprocess(self):
        '''
        Splits input texts into smaller overlapping chunks using LangChain's text splitter.
        '''
        logger.info('Chunking input documents with size %d and overlap %d', self.chunk_size, self.chunk_overlap)
        docs = [Document(page_content=text) for text in tqdm(self.texts, desc='Preparing documents')]
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '\n\n', '  \n'],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        self.processed_texts = [chunk.page_content for chunk in tqdm(chunks, desc='Splitting documents')]
        logger.info('Generated %d total chunks from %d original texts', len(self.processed_texts), len(self.texts))

    def _train_model(self):
        '''
        Trains the Top2Vec model on the preprocessed chunks.
        '''
        self.model = Top2Vec(self.processed_texts, speed=self.speed, workers=self.workers)

    def print_topics(self, num_topics: int = None):
        '''
        Prints the top keywords from the discovered topics.

        Args:
            num_topics: Number of top topics to display. If None, displays all.
        '''
        topic_count = self.model.get_num_topics() if num_topics is None else num_topics
        topic_words, _, topic_nums = self.model.get_topics(topic_count)
        for i, words in enumerate(topic_words):
            print(f'\nTopic #{topic_nums[i]}:')
            print(', '.join(words))

    def show_wordcloud(self, topic_id: int = 0):
        '''
        Displays a word cloud for a given topic ID.

        Args:
            topic_id: The topic index to visualize.
        '''
        self.model.generate_topic_wordcloud(topic_id)

    def reduce_topics(self, num_topics: int = 1):
        '''
        Reduces the number of topics using hierarchical topic reduction.

        Args:
            num_topics: The number of final reduced topics.
        '''
        logger.info('Reducing topics to %d using hierarchical topic reduction...', num_topics)
        self.model.hierarchical_topic_reduction(num_topics=num_topics)
        self.hierarchy = self.model.get_topic_hierarchy()
        for original_topic, reduced_topic in enumerate(self.hierarchy):
            print(f'Original Topic {original_topic} → Reduced Topic {reduced_topic}')
