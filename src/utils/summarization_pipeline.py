'''
Architecture Design:

We define a `SummarizationPipeline` class that wraps Hugging Face
Seq2Seq models (e.g. BART, T5, Pegasus) or custom local models. It provides:

1. **Model Management**
   - `load_model(model_name_or_path: str, in_8bit: bool=False)`
   - `save_model(save_path: str)`
   - `load_from_local(path: str)`

2. **Tokenization & Preprocessing**
   - `preprocess(texts: List[str], summaries: Optional[List[str]] =
        None, max_source_length: int, max_target_length: int)`
   - `tokenize_dataset(dataset: Dataset)`

3. **Training & Fine-Tuning**
   - `init_trainer(training_args: Seq2SeqTrainingArguments, train_dataset, eval_dataset)`
   - `train()`

4. **Generation & Evaluation**
   - `summarize(texts: List[str], max_new_tokens: int, **generate_kwargs)`
   - `compute_metrics(predictions, references)`

5. **Utilities**
   - `set_device(device: str)`
   - `update_training_args(**kwargs)`

Additional: integrated logging for step-by-step trace.
'''

import os
import logging
from typing import List, Union, Optional
from dataclasses import dataclass, field

import numpy as np
import torch
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)
from utils import setup_logger


@dataclass
class ModelConfig:
    '''Model configuration properties.'''

    model_name_or_path: str
    device: Optional[Union[str, torch.device]] = None
    in_8bit: bool = False
    is_pegasus: bool = False
    prefix: Optional[str] = None


@dataclass
class LoggingConfig:
    '''Logging configuration properties.'''

    log_file: Optional[str] = None
    log_level: int = logging.INFO


@dataclass
class SummarizationPipeline:
    '''
    Generic Summarizer Class for different models:
        - Loading from HuggingFace
        - Training
        - Local loading
    Supports:
        - BART
        - T5
        - PEGASUS (XSUM and human-centered)
    '''

    model_config: ModelConfig
    logging_config: LoggingConfig

    logger: logging.Logger = field(init=False)
    tokenizer: Optional[object] = field(init=False, default=None)
    model: Optional[object] = field(init=False, default=None)
    model_max_length: Optional[int] = field(init=False, default=None)
    trainer: Optional[object] = field(init=False, default=None)
    prefix: str = field(init=False)
    device: Union[str, torch.device] = field(init=False)

    def __post_init__(self):
        self.logger = setup_logger(
            name=self.__class__.__name__,
            log_file=self.logging_config.log_file,
            level=self.logging_config.log_level,
        )
        self.logger.info(
            'Initializing pipeline with model %s', self.model_config.model_name_or_path
        )

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = self.model_config.device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_config.is_pegasus = 'pegasus' in self.model_config.model_name_or_path.lower()
        self.prefix = (
            self.model_config.prefix
            if self.model_config.prefix is not None
            else ('' if self.model_config.is_pegasus else 'summarize: ')
        )

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        '''Load the appropriate model and tokenizer based on model type.'''
        self.logger.info('Loading tokenizer for %s', self.model_config.model_name_or_path)

        if self.model_config.is_pegasus:
            self.logger.info('Using specialized PegasusTokenizer')
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_config.model_name_or_path)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                self.model_config.model_name_or_path,
                load_in_8bit=self.model_config.in_8bit,
                device_map='auto' if str(self.model_config.device).startswith('cuda') else None,
            ).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name_or_path, use_fast=True
            )
            self.logger.info(
                'Loading model for %s (8bit=%s, device=%s)',
                self.model_config.model_name_or_path,
                self.model_config.in_8bit,
                self.device,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config.model_name_or_path,
                load_in_8bit=self.model_config.in_8bit,
                device_map='auto' if str(self.device).startswith('cuda') else None,
            ).to(self.device)

        self.model_max_length = self.tokenizer.model_max_length
        self.logger.info(
            'Model and tokenizer loaded successfully. Model max length: %s', self.model_max_length
        )
        self.logger.info('Using prefix: %s', self.prefix)

        model_vocab_size = self.model.config.vocab_size
        tokenizer_vocab_size = len(self.tokenizer)

        if model_vocab_size != tokenizer_vocab_size:
            self.logger.warning(
                f"Model vocab size ({model_vocab_size}) and tokenizer vocab size ({tokenizer_vocab_size}) mismatch. Resizing model embeddings."
            )
            self.model.resize_token_embeddings(tokenizer_vocab_size)
        else:
            self.logger.info("Model and tokenizer vocab sizes match. No resizing needed.")

    def set_device(self, device: Union[str, torch.device]):
        '''Setting device as hardware parameter.'''
        self.logger.info('Setting device to %s', device)
        self.device = device
        self.model.to(device)

    def preprocess(
        self,
        texts: List[str],
        summaries: Optional[List[str]] = None,
        max_target_length: int = 128,
        max_source_length: int = None,
    ):
        '''Pre-process (tokenize) dataset for fine-tuning.'''
        if not max_source_length:
            max_source_length = min(self.model_max_length, 4096)

        self.logger.debug(
            'Preprocessing %d texts (max_source_length=%d, max_target_length=%d)',
            len(texts),
            max_source_length,
            max_target_length,
        )

        inputs = [self.prefix + t for t in texts] if self.prefix else texts

        if self.model_config.is_pegasus:
            model_inputs = self.tokenizer(inputs, max_length=max_source_length, truncation=True)
        else:
            model_inputs = self.tokenizer(
                inputs, max_length=max_source_length, truncation=True, padding='max_length'
            )

        if summaries is not None:
            if self.model_config.is_pegasus:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(
                        summaries, max_length=max_target_length, truncation=True
                    )
            else:
                labels = self.tokenizer(
                    summaries, max_length=max_target_length, truncation=True, padding='max_length'
                )

            model_inputs['labels'] = labels['input_ids']
            self.logger.debug('Encoded labels for %s summaries', len(summaries))

        return model_inputs

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = 'text',
        summary_column: str = 'summary',
        batched: bool = True,
    ) -> Dataset:
        '''Tokenize a dataset for training or evaluation'''
        self.logger.info(
            'Tokenizing dataset with columns: text=%s, summary=%s', text_column, summary_column
        )

        def _tokenize_function(examples):
            return self.preprocess(examples[text_column], examples.get(summary_column, None))

        return dataset.map(_tokenize_function, batched=batched)

    def init_trainer(
        self,
        training_args: Seq2SeqTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        data_collator=None,
    ):
        '''Initialize the trainer with datasets and configuration'''
        self.logger.info('Initializing Seq2SeqTrainer')

        if data_collator is None:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer, model=self.model, padding='max_length'
            )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        self.logger.info('Trainer initialized.')

    def train(self):
        '''Train the model using the initialized trainer'''
        self.logger.info('Starting training...')
        assert self.trainer is not None, 'Trainer not initialized.'
        self.trainer.train()
        self.logger.info('Training complete.')

    def summarize(
        self, texts: List[str], max_new_tokens: int = 100, min_length: int = 10, **generate_kwargs
    ) -> List[str]:
        '''Generate summaries using the model'''
        self.logger.info(
            'Generating summaries for %s texts (max_new_tokens= %s)', len(texts), max_new_tokens
        )

        inputs_text = [self.prefix + t for t in texts] if self.prefix else texts

        if self.model_config.is_pegasus:
            inputs = self.tokenizer(
                inputs_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=min(self.model_max_length, 1024),
            ).to(self.device)

            # Default generation parameters optimized for PEGASUS
            pegasus_params = {
                'num_beams': 8,
                'length_penalty': 0.8,
                'no_repeat_ngram_size': 3,
                'early_stopping': True,
                'min_length': min_length,
            }

            # Override with any user-provided parameters
            generation_params = {**pegasus_params, **generate_kwargs}

            outputs = self.model.generate(**inputs, max_length=max_new_tokens, **generation_params)
        else:
            inputs = self.tokenizer(
                inputs_text, return_tensors='pt', padding='max_length', truncation=True
            ).to(self.device)
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, **generate_kwargs
            )

        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.logger.info('Summary generation complete.')
        return summaries

    def compute_metrics(self, eval_pred):
        '''Compute ROUGE metrics for evaluation'''
        self.logger.debug('Computing metrics...')
        preds, labels = eval_pred

        logger.debug(f"preds shape: {preds.shape}, max token id in preds: {preds.max()}")

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = evaluate.load('rouge').compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )

        gen_lens = [(pred != self.tokenizer.pad_token_id).sum() for pred in preds]
        result['gen_len'] = np.mean(gen_lens)

        self.logger.debug('Metrics computed: %s', result)
        return {k: round(v, 4) for k, v in result.items()}

    def save_model(self, save_path: str):
        '''Save the model and tokenizer to disk'''
        self.logger.info('Saving model and tokenizer to %s', save_path)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info('Save complete.')

    def load_from_local(self, path: str):
        '''Load a saved model from disk'''
        self.logger.info('Loading model and tokenizer from local path: %s', path)

        if 'pegasus' in path.lower() or self.model_config.is_pegasus:
            self.model = PegasusForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.model_config.is_pegasus = 'pegasus' in path.lower()
        self.logger.info(
            'Local load complete. Model is %s type',
            'pegasus' if self.model_config.is_pegasus else 'standard',
        )
