'''State-of-the-art hierarchical summarization with Huggingface models'''

# In[ ]:


import os
import gc
from datetime import date

import torch
from tqdm.notebook import tqdm
import pandas as pd

from utils import (
    compute_metrics,
    load_all_available_transcripts,
    SummarizationPipeline,
    TextChunker,
    LoggingConfig,
    ModelConfig,
)


# In[ ]:


transcripts_df = load_all_available_transcripts()
transcripts_df.shape


# In[ ]:


original_texts = transcripts_df['full_text'].tolist()
metadata = transcripts_df[['uuid', 'companyid', 'companyname', 'word_count_nltk']]


# In[ ]:


checkpoints = [
    'facebook/bart-large-cnn',
    'google-t5/t5-base',
    'google/pegasus-x-large',
    'human-centered-summarization/financial-summarization-pegasus',
]

all_metrics = []


# In[ ]:


for checkpoint in checkpoints:
    # Initialize model and pipeline
    model_config = ModelConfig(
        model_name_or_path=checkpoint, device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    pipeline = SummarizationPipeline(model_config=model_config, logging_config=LoggingConfig())

    tokenizer = pipeline.get_tokenizer()
    chunker = TextChunker(tokenizer)

    # Generate summaries for every transcript
    summaries = []
    for text in tqdm(original_texts, desc=f'Summarizing with {checkpoint}'):
        # Break text into manageable chunks
        chunks = chunker.chunk_text(text)
        # Summarize each chunk
        chunk_summaries = [pipeline.summarize(c) for c in chunks]
        combined = ' '.join(chunk_summaries)

        # If combined summary is too long, iteratively re-chunk and re-summarize
        max_rounds = 5
        for _ in range(max_rounds):
            tokens = tokenizer(combined, return_tensors='pt', truncation=False)['input_ids']
            if tokens.shape[1] <= min(1024, pipeline.model_max_length):
                break
            re_chunks = chunker.chunk_text(combined)
            combined = ' '.join(pipeline.summarize(rc) for rc in re_chunks)

        summaries.append(combined)

    # Clean up GPU memory
    del pipeline, model_config
    torch.cuda.empty_cache()
    gc.collect()

    # Compute metrics for this model's summaries
    metrics_df = compute_metrics(
        originals=original_texts,
        summaries=summaries,
        model_name=checkpoint,
        summarization_type='baseline',
    )

    metrics_df['uuid'] = metadata['uuid'].values
    metrics_df['companyid'] = metadata['companyid'].values
    metrics_df['companyname'] = metadata['companyname'].values
    metrics_df['wor_count_nltk'] = metadata['word_count_nltk'].values
    metrics_df['summary'] = summaries
    metrics_df['evaluation_date'] = date.today().isoformat()

    all_metrics.append(metrics_df)

# Concatenate all model metrics into one DataFrame
final_df = pd.concat(all_metrics, ignore_index=True)


# In[ ]:


output_path = 'summarization_evaluation_metrics.csv'
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
print(f'State-of-the-art evaluation complete. Metrics saved to {output_path}.')
