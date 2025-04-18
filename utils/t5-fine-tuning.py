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
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class SummarizationPipeline:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[Union[str, torch.device]] = None,
        in_8bit: bool = False,
        prefix: str = "summarize: "  # default prefix for models like BART/T5
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing pipeline with model {model_name_or_path}")

        self.model_name_or_path = model_name_or_path
        self.in_8bit = in_8bit

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix = prefix
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.rouge = evaluate.load("rouge")

        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        self.logger.info(f"Loading tokenizer for {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=True
        )
        self.logger.info(f"Loading model for {self.model_name_or_path}"
                         f" (8bit={self.in_8bit}, device={self.device})")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name_or_path,
            load_in_8bit=self.in_8bit,
            device_map="auto" if str(self.device).startswith("cuda") else None
        ).to(self.device)
        self.logger.info("Model and tokenizer loaded successfully.")

    def set_device(self, device: Union[str, torch.device]):
        self.logger.info(f"Setting device to {device}")
        self.device = device
        self.model.to(device)

    def preprocess(
        self,
        texts: List[str],
        summaries: Optional[List[str]] = None,
        max_source_length: int = 1024,
        max_target_length: int = 128
    ):
        self.logger.debug(f"Preprocessing {len(texts)} texts (max_source_length={max_source_length}, max_target_length={max_target_length})")
        inputs = [self.prefix + t for t in texts]
        model_inputs = self.tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length"
        )
        if summaries is not None:
            labels = self.tokenizer(
                summaries,
                max_length=max_target_length,
                truncation=True,
                padding="max_length"
            )
            model_inputs["labels"] = labels["input_ids"]
            self.logger.debug(f"Encoded labels for {len(summaries)} summaries")
        return model_inputs

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        summary_column: str = "summary",
        batched: bool = True
    ) -> Dataset:
        self.logger.info(f"Tokenizing dataset with columns: text='{text_column}', summary='{summary_column}'")
        return dataset.map(
            lambda examples: self.preprocess(
                examples[text_column],
                examples.get(summary_column, None)
            ),
            batched=batched
        )

    def init_trainer(
        self,
        training_args: Seq2SeqTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset
    ):
        self.logger.info("Initializing Seq2SeqTrainer")
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        self.logger.info("Trainer initialized.")

    def train(self):
        self.logger.info("Starting training...")
        assert self.trainer is not None, "Trainer not initialized."
        self.trainer.train()
        self.logger.info("Training complete.")

    def summarize(
        self,
        texts: List[str],
        max_new_tokens: int = 100,
        **generate_kwargs
    ) -> List[str]:
        self.logger.info(f"Generating summaries for {len(texts)} texts (max_new_tokens={max_new_tokens})")
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            **generate_kwargs
        )
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        self.logger.info("Summary generation complete.")
        return summaries

    def compute_metrics(self, eval_pred):
        self.logger.debug("Computing metrics...")
        preds, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(
            preds, skip_special_tokens=True
        )
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True
        )
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        gen_lens = [
            (pred != self.tokenizer.pad_token_id).sum() for pred in preds
        ]
        result["gen_len"] = np.mean(gen_lens)
        self.logger.debug(f"Metrics computed: {result}")
        return {k: round(v, 4) for k, v in result.items()}

    def save_model(self, save_path: str):
        self.logger.info(f"Saving model and tokenizer to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.logger.info("Save complete.")

    def load_from_local(self, path: str):
        self.logger.info(f"Loading model and tokenizer from local path: {path}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.logger.info("Local load complete.")

# ---------- Example: Fine-tuning on BillSum Dataset ----------
if __name__ == "__main__":
    checkpoint = 'google-t5/t5-base'

    pipeline = SummarizationPipeline(
        model_name_or_path=checkpoint,
        in_8bit=False
    )

    billsum = load_dataset("FiscalNote/billsum")
    tokenized = pipeline.tokenize_dataset(billsum["train"])
    tokenized_eval = pipeline.tokenize_dataset(billsum["test"])

    training_args = Seq2SeqTrainingArguments(
        output_dir="../models/ragsum-t5-billsum",
        eval_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        warmup_steps=100,
        max_steps=50,
        fp16=True,
        logging_steps=8,
    )

    pipeline.init_trainer(training_args, tokenized, tokenized_eval)
    pipeline.train()
    pipeline.save_model("../models/ragsum-t5-billsum")

    sample_text = [
        "Deep Learning models have achieved state-of-the-art performance across various NLP benchmarks. However, summarizing long documents remains a challenge due to limited context windows and dependencies on fine-grained linguistic information."
    ]

    summary = pipeline.summarize(sample_text, max_new_tokens=60)
    print(f"{checkpoint} fine-tuned summary:", summary[0])
