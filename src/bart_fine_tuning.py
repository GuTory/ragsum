"""Fine-tuning script for BART model."""

import os
import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

from datasets import load_dataset
from utils import setup_logger

logger = setup_logger(__name__)
logger.info('Logging Successfully set up')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
rouge = evaluate.load('rouge')

checkpoint = 'facebook/bart-large-cnn'
tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    device_map='auto',
)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

billsum = load_dataset('FiscalNote/billsum')

prefix = 'summarize: '


def preprocess_function(examples):
    '''Preprocessing dataset.'''
    inputs = [prefix + doc for doc in examples['text']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples['summary'], max_length=128, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


tokenized_billsum = billsum.map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    '''Metric computation function.'''
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result['gen_len'] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


seq2seq_args = Seq2SeqTrainingArguments(
    output_dir=f'../models/ragsum-{checkpoint}-billsum',
    eval_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=6,
    predict_with_generate=True,
    warmup_steps=100,
    max_steps=200,
    fp16=True,
    logging_steps=32,
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

trainer = Seq2SeqTrainer(
    model=model,
    args=seq2seq_args,
    train_dataset=tokenized_billsum['train'],
    eval_dataset=tokenized_billsum['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

save_path = f'../models/ragsum-{checkpoint}-billsum'
os.makedirs(save_path, exist_ok=True)
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)


def test_model():
    '''Model testing function for showcasing measurements.'''
    saved_model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
    saved_tokenizer = AutoTokenizer.from_pretrained(save_path)

    test_text = '''
    summarize: The United States Congress has approved a sweeping infrastructure bill, marking one of the most significant legislative efforts in recent years. The bill, which totals $1.2 trillion in funding, aims to revamp aging infrastructure nationwide. Key areas of investment include transportation — such as roads, railways, and bridges — clean energy initiatives, and expanded broadband internet access. The legislation received bipartisan support in both the House and Senate, signaling rare political cooperation in an otherwise divided climate. Proponents argue that this investment will create jobs, stimulate the economy, and lay the groundwork for long-term national competitiveness.
    '''

    inputs = tokenizer(test_text, return_tensors='pt').input_ids
    outputs = saved_model.generate(inputs, max_new_tokens=100, do_sample=False)
    summary = saved_tokenizer.decode(outputs[0], skip_special_tokens=True)

    reference_summary = '''
    Congress passed a new bill aimed at improving infrastructure across the U.S., allocating $1.2 trillion in funding over the next ten years. The legislation focuses on roads, bridges, clean energy, and broadband access, with bipartisan support marking a significant political achievement.
    '''

    results = rouge.compute(predictions=[summary], references=[reference_summary], use_stemmer=True)
    results = {k: round(v, 4) for k, v in results.items()}

    print('Generated summary:', summary)
    print('ROUGE scores:', results)


if __name__ == '__main__':
    test_model()
