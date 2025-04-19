"""
Using SummarizationPipeline to fine-tune BART
"""

from utils import SummarizationPipeline
from datasets import load_dataset

from transformers import Seq2SeqTrainingArguments


if __name__ == "__main__":
    checkpoint = 'google-t5/t5-base'

    pipeline = SummarizationPipeline(model_name_or_path=checkpoint, in_8bit=False)

    billsum = load_dataset("FiscalNote/billsum")
    tokenized = pipeline.tokenize_dataset(billsum["train"])
    tokenized_eval = pipeline.tokenize_dataset(billsum["test"])

    save_path = f"../models/ragsum-{checkpoint}-billsum"
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
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
    pipeline.save_model(save_path)

    sample_text = [
        "Deep Learning models have achieved state-of-the-art performance across various NLP benchmarks. However, summarizing long documents remains a challenge due to limited context windows and dependencies on fine-grained linguistic information."
    ]

    summary = pipeline.summarize(sample_text, max_new_tokens=60)
    print(f"{checkpoint} fine-tuned summary:", summary[0])
