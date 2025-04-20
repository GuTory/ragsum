'''Retriever class for RAG capabilities.'''

import numpy as np
from typing import List, TypeAlias
from dataclasses import dataclass, field
import faiss
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import setup_logger

logger = setup_logger(__name__)

SearchResults: TypeAlias = tuple[list[str], list[float]]


@dataclass
class Retriever:
    '''Retriever class for RAG systems.'''

    knowledge: List[str]
    top_k: int = 3
    type: str = 'float32'
    model: SentenceTransformer = field(
        default_factory=lambda: SentenceTransformer('all-MiniLM-L6-v2')
    )
    embeddings: np.ndarray = field(init=False)
    index: faiss.IndexFlatL2 = field(init=False)
    distances: np.ndarray = field(init=False)
    indices: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.top_k <= 0:
            raise ValueError('Give a positive top k parameter.')

        self.embeddings = np.array(
            [
                self.model.encode([text])[0]
                for text in tqdm(self.knowledge, desc="Encoding", unit="doc")
            ]
        ).astype(self.type)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        with tqdm(total=self.embeddings.shape[0], desc="Indexing", unit="vec") as pbar:
            for i in range(0, self.embeddings.shape[0], 1000):
                batch = self.embeddings[i : i + 1000]
                self.index.add(batch)
                pbar.update(batch.shape[0])

        self.distances = np.empty(0)
        self.indices = np.empty(0)

    def search(self, query: str, top_k: int = None) -> SearchResults:
        '''Search the query in the index by similarity'''
        if not top_k:
            top_k = self.top_k
        query_embedding = self.model.encode([query]).astype(self.type)
        distances, indices = self.index.search(query_embedding, top_k)
        self.distances = distances[0]
        self.indices = indices[0]

        def decode_indices(indices, texts):
            '''
            Given a list/array of indices and a mapping list (texts),
            return the decoded text entries.
            '''
            return [texts[i] for i in indices]

        return decode_indices(self.indices, self.knowledge), self.distances


if __name__ == '__main__':
    df = pd.read_csv(
        'hf://datasets/sohomghosh/FinRAD_Financial_Readability_Assessment_Dataset/FinRAD_13K_terms_definitions_labels.csv'
    )
    df = df[['terms', 'definitions', 'source', 'assigned_readability']]
    df = df.dropna(subset=['definitions'])
    df['combined'] = df['terms'] + ': ' + df['definitions']

    df.head(10)

    retriever = Retriever(df.combined.tolist(), 5)

    top_k, distances = retriever.search(
        'What is the meaning of arbitrage in general? and tell me about acquisitions', top_k=5
    )

    for k, dist in zip(top_k, distances):
        print(f'distance: [{dist:.4f}]: {k}')
