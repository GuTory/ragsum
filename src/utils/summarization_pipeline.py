"""
Architecture Design:

We define a `SummarizationPipeline` class that wraps Hugging Face Seq2Seq models (e.g. BART, T5, Pegasus) or custom local models. It provides:

1. **Model Management**
   - `load_model(model_name_or_path: str, in_8bit: bool=False)`
   - `save_model(save_path: str)`
   - `load_from_local(path: str)`

2. **Tokenization & Preprocessing**
   - `preprocess(texts: List[str], summaries: Optional[List[str]] = None, max_source_length: int, max_target_length: int)`
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
"""

import os
import logging
from typing import List, Union, Optional

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


class SummarizationPipeline:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        in_8bit: bool = False,
        # Empty default prefix for Pegasus models
        prefix: Optional[str] = None,
        log_file: str = None,
        log_level: int = logging.INFO,
    ):
        self.logger = setup_logger(name=self.__class__.__name__, log_file=log_file, level=log_level)
        self.logger.info(f"Initializing pipeline with model {model_name_or_path}")

        self.model_name_or_path = model_name_or_path
        self.in_8bit = in_8bit

        # Handle device setup
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set prefix based on model type
        self.is_pegasus = "pegasus" in model_name_or_path.lower()
        # If prefix is None, set appropriate default based on model type
        if prefix is None:
            self.prefix = "" if self.is_pegasus else "summarize: "
        else:
            self.prefix = prefix

        self.tokenizer = None
        self.model = None
        self.model_max_length = None
        self.trainer = None
        self.rouge = evaluate.load("rouge")

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """Load the appropriate model and tokenizer based on model type"""
        self.logger.info(f"Loading tokenizer for {self.model_name_or_path}")

        if self.is_pegasus:
            self.logger.info("Using specialized PegasusTokenizer")
            self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name_or_path)
            self.model = PegasusForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                load_in_8bit=self.in_8bit,
                device_map="auto" if str(self.device).startswith("cuda") else None,
            ).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
            self.logger.info(
                f"Loading model for {self.model_name_or_path}" f" (8bit={self.in_8bit}, device={self.device})"
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name_or_path,
                load_in_8bit=self.in_8bit,
                device_map="auto" if str(self.device).startswith("cuda") else None,
            ).to(self.device)

        self.model_max_length = self.tokenizer.model_max_length
        self.logger.info(f"Model and tokenizer loaded successfully. Model max length: {self.model_max_length}")
        self.logger.info(f"Using prefix: '{self.prefix}'")

    def set_device(self, device: Union[str, torch.device]):
        self.logger.info(f"Setting device to {device}")
        self.device = device
        self.model.to(device)

    def preprocess(
        self,
        texts: List[str],
        summaries: Optional[List[str]] = None,
        max_target_length: int = 128,
        max_source_length: int = None,
    ):
        if not max_source_length:
            max_source_length = self.model_max_length

        self.logger.debug(
            f"Preprocessing {len(texts)} texts (max_source_length={max_source_length}, max_target_length={max_target_length})"
        )

        # Apply prefix only if it's defined and non-empty
        inputs = [self.prefix + t for t in texts] if self.prefix else texts

        if self.is_pegasus:
            model_inputs = self.tokenizer(inputs, max_length=max_source_length, truncation=True)
        else:
            model_inputs = self.tokenizer(inputs, max_length=max_source_length, truncation=True, padding="max_length")

        if summaries is not None:
            if self.is_pegasus:
                with self.tokenizer.as_target_tokenizer():
                    labels = self.tokenizer(summaries, max_length=max_target_length, truncation=True)
            else:
                labels = self.tokenizer(summaries, max_length=max_target_length, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            self.logger.debug(f"Encoded labels for {len(summaries)} summaries")

        return model_inputs

    def tokenize_dataset(
        self, dataset: Dataset, text_column: str = "text", summary_column: str = "summary", batched: bool = True
    ) -> Dataset:
        """Tokenize a dataset for training or evaluation"""
        self.logger.info(f"Tokenizing dataset with columns: text='{text_column}', summary='{summary_column}'")

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
        """Initialize the trainer with datasets and configuration"""
        self.logger.info("Initializing Seq2SeqTrainer")

        if data_collator is None:
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding="max_length")

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        self.logger.info("Trainer initialized.")

    def train(self):
        """Train the model using the initialized trainer"""
        self.logger.info("Starting training...")
        assert self.trainer is not None, "Trainer not initialized."
        self.trainer.train()
        self.logger.info("Training complete.")

    def summarize(
        self, texts: List[str], max_new_tokens: int = 100, min_length: int = 10, **generate_kwargs
    ) -> List[str]:
        """Generate summaries using the model"""
        self.logger.info(f"Generating summaries for {len(texts)} texts (max_new_tokens={max_new_tokens})")

        inputs_text = [self.prefix + t for t in texts] if self.prefix else texts

        if self.is_pegasus:
            inputs = self.tokenizer(
                inputs_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=min(self.model_max_length, 1024),
            ).to(self.device)

            # Default generation parameters optimized for PEGASUS
            pegasus_params = {
                "num_beams": 8,
                "length_penalty": 0.8,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "min_length": min_length,
            }

            # Override with any user-provided parameters
            generation_params = {**pegasus_params, **generate_kwargs}

            outputs = self.model.generate(**inputs, max_length=max_new_tokens, **generation_params)
        else:
            inputs = self.tokenizer(inputs_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_kwargs)

        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.logger.info("Summary generation complete.")
        return summaries

    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics for evaluation"""
        self.logger.debug("Computing metrics...")
        preds, labels = eval_pred

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        gen_lens = [(pred != self.tokenizer.pad_token_id).sum() for pred in preds]
        result["gen_len"] = np.mean(gen_lens)

        self.logger.debug(f"Metrics computed: {result}")
        return {k: round(v, 4) for k, v in result.items()}

    def save_model(self, save_path: str):
        """Save the model and tokenizer to disk"""
        self.logger.info(f"Saving model and tokenizer to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info("Save complete.")

    def load_from_local(self, path: str):
        """Load a saved model from disk"""
        self.logger.info(f"Loading model and tokenizer from local path: {path}")

        if "pegasus" in path.lower() or self.is_pegasus:
            self.model = PegasusForConditionalGeneration.from_pretrained(path).to(self.device)
            self.tokenizer = PegasusTokenizer.from_pretrained(path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(path)

        self.is_pegasus = "pegasus" in path.lower()
        self.logger.info(f"Local load complete. Model is {'PEGASUS' if self.is_pegasus else 'standard'} type.")
