'''Zero-shot topic modeling Modeler class.'''

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from top2vec import Top2Vec
from langchain.schema import Document
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class TopicModeler:
    '''
    TopicModeler wraps Top2Vec for pre-chunked Document inputs.

    Example:
        from langchain.schema import Document

        chunks = [Document(page_content='chunk1 text...'), Document(page_content='chunk2 text...')]
        tm = TopicModeler(chunks=chunks , speed='learn', workers=8)
        tm.print_topics()
        tm.generate_wordcloud(0)
    '''

    chunks: list
    speed: str = 'learn'
    workers: int = 4
    num_topics: int = field(init=False)

    def __post_init__(self):
        '''
        Train Top2Vec on pre-chunked Documents.

        Args:
            chunks: List of Document objects, each with .page_content set to text chunk.
        '''
        self.model: Optional[Top2Vec] = None
        texts = [chunk.page_content for chunk in self.chunks]
        logger.info(
            f'Training Top2Vec on {len(texts)} chunks '
            f'(speed={self.speed}, workers={self.workers})…'
        )
        self.model = Top2Vec(texts, speed=self.speed, workers=self.workers)
        self.model.hierarchical_topic_reduction(num_topics=1)

        self.get_num_topics()
        logger.info('Model training complete.')

    def get_num_topics(self):
        if self.model is None:
            raise ValueError('Model has not been trained. Call fit() first.')
        self.num_topics = self.model.get_num_topics()
        return self.num_topics

    def get_topics(
        self, num_topics: Optional[int]
    ) -> Tuple[List[List[str]], List[List[float]], List[int]]:
        '''
        Retrieve topics from the trained model.

        Returns:
            topic_words: top words per topic
            word_scores: scores for each word
            topic_nums: topic identifiers
        '''
        if self.model is None:
            raise ValueError('Model has not been trained. Call fit() first.')
        if not num_topics:
            num_topics = self.num_topics
        return self.model.get_topics(num_topics)

    def print_topics(self) -> None:
        '''
        Print the top N words for each topic.
        '''
        topic_words, _, topic_nums = self.get_topics()
        for words, tid in zip(topic_words, topic_nums):
            print(f'Topic #{tid}: ' + ', '.join(words))

    def generate_wordcloud(self, topic_num: int) -> None:
        '''
        Generate and show a wordcloud for a given topic.

        Args:
            topic_num: The numeric ID of the topic.
        '''
        if self.model is None:
            raise ValueError('Model has not been trained. Call fit() first.')
        logger.info(f'Generating wordcloud for topic {topic_num}…')
        self.model.generate_topic_wordcloud(topic_num)
