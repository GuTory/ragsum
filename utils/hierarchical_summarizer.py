'''
Module for an off-the-shelf hierarchical
summarization pipeline with a selected model.
'''

import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Literal, List, Optional
import logging
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    AutoModelForSeq2SeqLM,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import bert_score
import textstat
import pandas as pd


class BaseSummarizer(ABC):
    '''
    Abstract base class for text summarization models.
    '''

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.max_length = max_length or self.tokenizer.model_max_length

        logging.basicConfig(level=logging.INFO, force=True)
        if torch.cuda.is_available():
            logging.info('GPU Name: %s', torch.cuda.get_device_name(0))
            logging.info('CUDA Version: %s', torch.version.cuda)
            logging.info(
                'GPU Memory Allocated: %.2f GB',
                torch.cuda.memory_allocated(0) / 1024**3,
            )
            logging.info('GPU Memory Reserved: %.2f GB', torch.cuda.memory_reserved(0) / 1024**3)

    @abstractmethod
    def _load_tokenizer(self):
        '''Load the appropriate tokenizer for the model.'''
        

    @abstractmethod
    def _load_model(self):
        '''Load the appropriate model.'''
        

    @abstractmethod
    def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        '''Generate a summary for the given text.'''
        


class PegasusSummarizer(BaseSummarizer):
    '''
    Implementation of the Pegasus model for text summarization.
    '''

    def _load_tokenizer(self):
        return PegasusTokenizer.from_pretrained(self.model_name)

    def _load_model(self):
        return PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)

    def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        inputs = self.tokenizer(
            text, return_tens='pt', truncation=True, max_length=self.max_length
        ).to(self.device)

        summary_ids = self.model.generate(
            **inputs, max_length=max_length or self.max_length // 4, early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class T5Summarizer(BaseSummarizer):
    '''
    Implementation of the T5 model for text summarization.
    '''

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def generate_summary(self, text: str, max_length: Optional[int] = None) -> str:
        inputs = self.tokenizer(
            'summarize: ' + text,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        summary_ids = self.model.generate(
            **inputs, max_length=max_length or self.max_length // 4, early_stopping=True
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class HierarchicalSummarizationPipeline:
    '''
    Class for an off-the-shelf hierarchical
    summarization pipeline with a selected model.
    '''

    SUMMARIZER_TYPES = {
        'pegasus': PegasusSummarizer,
        't5': T5Summarizer,
    }

    # Predefined model configurations
    MODEL_CONFIGS = {
        'pegasus': {
            'default': 'google/pegasus-xsum',
            'financial': 'human-centered-summarization/financial-summarization-pegasus',
            'news': 'google/pegasus-newsroom',
            'multi_news': 'google/pegasus-multi_news',
        },
        't5': {
            'default': 't5-base',
            'large': 't5-large',
            'small': 't5-small',
            'news': 't5-base-news-summarization',
        },
    }

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str,
        summarization_type: Literal['baseline', 'RAG', 'fine-tuned', 'fine-tuned RAG'],
        model_type: str = 'pegasus',
    ):
        self.df = df
        self.model_name = model_name
        self.summarization_type = summarization_type

        if model_type not in self.SUMMARIZER_TYPES:
            raise ValueError(
                f'Unsupported model type: {model_type}.'
                f'Supported types: {list(self.SUMMARIZER_TYPES.keys())}'
            )

        self.summarizer = self.SUMMARIZER_TYPES[model_type](
            model_name=model_name, max_length=None  # Will use model's default
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.summarizer.max_length,
            chunk_overlap=self.summarizer.max_length / 10,
        )

    def __repr__(self):
        return (
            f'HierarchicalSummarizationPipeline(model_name={self.model_name}, '
            f'summarization_type={self.summarization_type}, device={self.summarizer.device}, '
            f'max_length={self.summarizer.max_length})'
        )

    def __str__(self):
        return (
            f'HierarchicalSummarizationPipeline with model {self.model_name} '
            f'using summarization type {self.summarization_type} '
            f'on device {self.summarizer.device} '
            f'with max token length {self.summarizer.max_length}'
        )

    def summarize_text(self, text: str) -> List[str]:
        '''
        Summarize a single text chunk using the model.

        Args:
            text (str): Input text to summarize

        Returns:
            List[str]: List of generated summaries
        '''
        chunks = self.text_splitter.split_text(text)
        logging.debug('Chunk lengths: %s', [len(c) for c in chunks])

        summaries = []
        for chunk in tqdm(chunks):
            summary = self.summarizer.generate_summary(chunk)
            summaries.append(summary)

        return summaries

    def recursive_summary(self, text: str, target_length: Optional[int] = None) -> str:
        '''
        Recursively summarize text until it reaches target length.

        Args:
            text (str): Input text to summarize
            target_length (int, optional): Target token length. Defaults to max_length.

        Returns:
            str: Final summarized text
        '''
        if target_length is None:
            target_length = self.summarizer.max_length

        tokens = self.summarizer.tokenizer.tokenize(text)
        logging.debug('Token size: %s', len(tokens))

        combined_summary = text
        while len(tokens) > target_length:
            summaries = self.summarize_text(combined_summary)
            combined_summary = ' '.join(summaries)
            tokens = self.summarizer.tokenizer.tokenize(combined_summary)

        return combined_summary

    def evaluate_summary(self, row: pd.Series) -> pd.DataFrame:
        '''
        Evaluate a single summary using various metrics.

        Args:
            row (pd.Series): DataFrame row containing text and summary

        Returns:
            pd.DataFrame: DataFrame with evaluation metrics
        '''
        text_to_summarize = row.full_text
        summary = row[f'{self.model_name}-summaries']
        uuid = row.uuid
        company_id = row.companyid
        company_name = row.companyname

        rouge_evaluator = Rouge()
        rouge_scores = rouge_evaluator.get_scores(summary, text_to_summarize)

        if isinstance(rouge_scores, list):
            rouge_scores = rouge_scores[0]

        reference_tokens = text_to_summarize.split()
        candidate_tokens = summary.split()
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens)

        p, r, f1 = bert_score.score(
            [summary], [text_to_summarize], rescale_with_baseline=True, lang='en'
        )

        original_len = len(text_to_summarize.split())
        summary_len = len(summary.split())
        compression_ratio = summary_len / original_len if original_len > 0 else 0

        readability = textstat.flesch_reading_ease(summary)

        results = {}
        results['model_name'] = self.model_name
        results['uuid'] = uuid
        results['companyid'] = company_id
        results['companyname'] = company_name

        for metric, scores in rouge_scores.items():
            results[f'{metric}_r'] = scores['r']
            results[f'{metric}_p'] = scores['p']
            results[f'{metric}_f'] = scores['f']

        results['bleu'] = bleu_score
        results['bert_precision'] = p.item()
        results['bert_recall'] = r.item()
        results['bert_f1'] = f1.item()
        results['compression_ratio'] = compression_ratio
        results['readability'] = readability

        return pd.DataFrame([results])

    def run_pipeline(self) -> pd.DataFrame:
        '''
        Run the complete summarization pipeline.

        Returns:
            pd.DataFrame: DataFrame with summaries and evaluation results
        '''
        # Generate summaries
        summaries = []
        for text in tqdm(self.df.full_text):
            summary = self.recursive_summary(text=text)
            summaries.append(summary)

        # Add summaries to DataFrame
        summary_column = f'{self.model_name}-summaries'
        self.df[summary_column] = summaries

        # Evaluate summaries
        evaluation_results = pd.DataFrame()
        for _, row in tqdm(self.df.iterrows()):
            evaluation_result = self.evaluate_summary(row)
            evaluation_results = pd.concat(
                [evaluation_results, evaluation_result], ignore_index=True
            )

        # Save results
        self._save_results(evaluation_results)

        return evaluation_results

    def _save_results(self, evaluation_results: pd.DataFrame):
        '''
        Save evaluation results to CSV file.

        Args:
            evaluation_results (pd.DataFrame): DataFrame containing evaluation results
        '''
        csv_filename = Path('..') / 'data' / 'evaluation_results.csv'

        if os.path.exists(csv_filename):
            existing_df = pd.read_csv(csv_filename)
            if (
                (existing_df.model_name == self.model_name)
                & (existing_df.companyid == self.df.companyid.iloc[0])
            ).any():
                logging.info(
                    'model %s and %s combination already exists in %s. no new row added.',
                    self.model_name,
                    self.df.companyid.iloc[0],
                    csv_filename,
                )
                updated_df = existing_df
            else:
                updated_df = pd.concat([existing_df, evaluation_results], ignore_index=True)
                logging.info(
                    'model %s not found. appending new row to %s.',
                    self.model_name,
                    csv_filename,
                )
        else:
            updated_df = evaluation_results
            logging.info('%s not found. creating new file.', csv_filename)

        updated_df.to_csv(csv_filename, index=False)
        logging.info('results saved to %s', csv_filename)

        # Save summaries
        self.df.to_csv(
            Path('..')
            / 'data'
            / 'summaries'
            / (f'{self.df.companyid.iloc[0]}_{self.model_name}.csv').replace('/', '-'),
            sep='\t',
            index=False,
            quoting=1,
            escapechar='\\',
            doublequote=True,
            quotechar='"',
            lineterminator='\n',
        )
