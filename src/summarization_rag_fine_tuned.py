'''Summarization with RAG and fine-tuned summarization models'''

import os
import gc
from datetime import date
from collections import Counter

import torch
from tqdm.notebook import tqdm
import pandas as pd
from langchain.schema import Document

from utils import (
    compute_metrics,
    load_all_available_transcripts,
    SummarizationPipeline,
    TextChunker,
    LoggingConfig,
    ModelConfig,
    Retriever,
    GensimTopicModeler as TopicModeler,
)

# Load transcripts
transcripts_df = load_all_available_transcripts()
print(transcripts_df.shape)

original_texts = transcripts_df['full_text'].tolist()
metadata = transcripts_df[['uuid', 'companyid', 'companyname', 'word_count_nltk']]

# Model setup
checkpoints = [
    'facebook/bart-large-cnn',
    'google-t5/t5-base',
    'google/pegasus-x-large',
    'human-centered-summarization/financial-summarization-pegasus',
]
local_paths = [f'../models/ragsum-{ckpt}-billsum' for ckpt in checkpoints]

logging_config = LoggingConfig()
all_metrics = []

# Load retriever data
df = pd.read_csv(
    'hf://datasets/sohomghosh/FinRAD_Financial_Readability_Assessment_Dataset/FinRAD_13K_terms_definitions_labels.csv'
)
df = df[['terms', 'definitions', 'source', 'assigned_readability']]
df = df.dropna(subset=['definitions'])
df['combined'] = df['terms'] + ': ' + df['definitions']
retriever = Retriever(df.combined.tolist(), 5)

# Loop through models
for checkpoint, path in zip(checkpoints, local_paths):
    model_config = ModelConfig(
        model_name_or_path=checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    pipeline = SummarizationPipeline(
        model_config=model_config, logging_config=logging_config, remote=False
    )
    pipeline.load_from_local(path)

    tokenizer = pipeline.get_tokenizer()
    chunker = TextChunker(tokenizer)

    summaries = []
    retrieved_counter = Counter()

    for i, text in tqdm(
        enumerate(original_texts),
        total=len(original_texts),
        desc=f'Fine-tuned summarizing with {checkpoint}',
    ):
        print(f'Summarizing text nr.{i}')
        chunks = chunker.chunk_text(text)

        tm = TopicModeler(chunks=[Document(page_content=doc) for doc in chunks], num_topics=6)
        topic_words, _, topic_nums = tm.get_topics(1)

        for words, tid in zip(topic_words, topic_nums):
            print(f'Topic: ' + ' '.join(words))

        topics_string = ' '.join(words)
        top_results, _ = retriever.search(topics_string, 3)

        # Update the counter with retrieved terms
        retrieved_counter.update(top_results)

        chunks.insert(0, 'context: ' + ', '.join(top_results) + '. Text to summarize: ')

        print(f'Inserted chunk: {chunks[0]}')

        chunk_summaries = [pipeline.summarize(c) for c in chunks]
        combined = ' '.join(chunk_summaries)

        # Iterative reduction if over length
        max_rounds = 5
        for _ in range(max_rounds):
            tokens = tokenizer(combined, return_tensors='pt', truncation=False)['input_ids']
            if tokens.shape[1] <= min(1024, pipeline.model_max_length):
                break
            re_chunks = chunker.chunk_text(combined)
            combined = ' '.join(pipeline.summarize(rc) for rc in re_chunks)

        summaries.append(combined)

    del pipeline, model_config
    torch.cuda.empty_cache()
    gc.collect()

    # Compute metrics
    metrics_df = compute_metrics(
        originals=original_texts,
        summaries=summaries,
        model_name=checkpoint,
        summarization_type='RAG + fine-tuned',
    )

    metrics_df['uuid'] = metadata['uuid'].values
    metrics_df['companyid'] = metadata['companyid'].values
    metrics_df['companyname'] = metadata['companyname'].values
    metrics_df['wor_count_nltk'] = metadata['word_count_nltk'].values
    metrics_df['summary'] = summaries
    metrics_df['evaluation_date'] = date.today().isoformat()

    all_metrics.append(metrics_df)

    # Print retriever usage stats for this model
    print(f'\nRetriever usage frequency for {checkpoint}:')
    for term, count in retrieved_counter.most_common():
        print(f'{term[:80]}... → {count} times')

# Combine all metrics
final_df = pd.concat(all_metrics, ignore_index=True)

# Save to CSV
output_path = 'summarization_evaluation_metrics_rag_ft.csv'
if os.path.exists(output_path):
    existing_df = pd.read_csv(
        output_path,
        sep='\t',
        quoting=1,
        quotechar='"',
        escapechar='\\',
        doublequote=True,
        engine='python',
    )
    final_df = pd.concat([existing_df, final_df], ignore_index=True)
final_df.to_csv(
    output_path,
    index=False,
    sep='\t',
    quoting=1,
    escapechar='\\',
    doublequote=True,
    quotechar='"',
)
print(f'\nFine-tuned model with RAG evaluation complete. Metrics saved to {output_path}.')
