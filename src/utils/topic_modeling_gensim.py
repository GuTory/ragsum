'''Topic modeling Class based on Gensim package.'''

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from langchain.schema import Document
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from utils import setup_logger

nltk.download('punkt')
nltk.download('stopwords')

logger = setup_logger(__name__)


@dataclass
class GensimTopicModeler:
    '''
    TopicModeler wraps Gensim for pre-chunked Document inputs.

    Example:
        from langchain.schema import Document

        chunks = [Document(page_content='chunk1 text...'), Document(page_content='chunk2 text...')]
        tm = GensimTopicModeler(chunks=chunks)
        topics = tm.get_topics(num_topics=5)
        tm.generate_wordcloud(0)
    '''

    chunks: List[Document]
    num_topics: int = 10
    passes: int = 10
    lda_model: Optional[models.LdaModel] = field(init=False)
    dictionary: Optional[corpora.Dictionary] = field(init=False)
    corpus: Optional[List[List[Tuple[int, int]]]] = field(init=False)

    def __post_init__(self):
        texts = [self._preprocess_text(doc.page_content) for doc in self.chunks]
        self.dictionary = corpora.Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda_model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
        )
        logger.info(f'Trained LDA model with {self.num_topics} topics.')

    def _preprocess_text(self, text: str) -> List[str]:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word.isalpha() and word not in stop_words]

    def get_topics(
        self, num_topics: Optional[int] = None
    ) -> Tuple[List[List[str]], List[List[float]], List[int]]:
        """
        Retrieve topics from the trained LDA model.

        Args:
            num_topics: Number of topics to retrieve (defaults to all if not specified).

        Returns:
            topic_words: top words per topic
            word_scores: probabilities for each word
            topic_nums: topic identifiers
        """
        if self.lda_model is None:
            raise ValueError('Model has not been trained.')

        if num_topics is None or num_topics > self.num_topics:
            num_topics = self.num_topics

        topic_words = []
        word_scores = []
        topic_nums = []

        for idx in range(num_topics):
            topic = self.lda_model.show_topic(idx, topn=10)
            words, scores = zip(*topic)
            topic_words.append(list(words))
            word_scores.append(list(scores))
            topic_nums.append(idx)

        return topic_words, word_scores, topic_nums

    def print_topics(self, num_words: int = 10):
        '''Printing the topics discovered in the text corpus.'''
        for idx, topic in self.lda_model.show_topics(
            num_topics=self.num_topics, num_words=num_words, formatted=False
        ):
            print(f"Topic #{idx}: {', '.join([word for word, _ in topic])}")

    def generate_wordcloud(self, topic_num: int):
        '''Visual generation of discovered words related to one topic.'''
        topic_words = dict(self.lda_model.show_topic(topic_num, topn=30))
        wc = WordCloud(width=800, height=400).generate_from_frequencies(topic_words)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic #{topic_num}")
        plt.show()
