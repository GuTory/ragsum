'''
Using SummarizationPipeline to fine-tune PEGASUS on billsum dataset
'''

from datasets import load_dataset
import torch
from transformers import Seq2SeqTrainingArguments

from utils import SummarizationPipeline, ModelConfig, LoggingConfig


if __name__ == '__main__':
    checkpoint = 'google/pegasus-xsum'

    model_config: ModelConfig = ModelConfig(
        model_name_or_path=checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    logging_config: LoggingConfig = LoggingConfig()

    pipeline = SummarizationPipeline(model_config=model_config, logging_config=logging_config)

    billsum = load_dataset('FiscalNote/billsum')

    tokenized = pipeline.tokenize_dataset(billsum['train'])
    tokenized_eval = pipeline.tokenize_dataset(billsum['test'])

    save_path = f'../models/ragsum-{checkpoint}-billsum'

    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,
        eval_strategy='epoch',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        warmup_steps=100,
        max_steps=50,
        fp16=True,
        logging_steps=8,
        generation_max_length=256,
        generation_num_beams=8,
    )

    pipeline.init_trainer(training_args, tokenized, tokenized_eval)
    pipeline.train()
    pipeline.save_model(save_path)

    sample_text = [
        'Deep Learning models have achieved state-of-the-art '
        'performance across various NLP benchmarks. However, '
        'summarizing long documents remains a challenge due to '
        'limited context windows and dependencies on '
        'fine-grained linguistic information.'
    ]

    summary = pipeline.summarize(sample_text, max_new_tokens=60, num_beams=8, length_penalty=0.8)
    print(f'{checkpoint} fine-tuned summary:', summary[0])
