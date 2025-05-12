'''Computing metrics for summarization.'''

import pandas as pd
from rouge import Rouge
import nltk

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
from nltk import word_tokenize
import textstat
import bert_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')


def compute_metrics(originals: list, summaries: list, model_name, summarization_type):
    '''Metric computation for summaries'''

    results = []
    rouge_evaluator = Rouge()

    for original_text, summary in zip(originals, summaries):
        uuid = company_id = company_name = None

        # ROUGE requires raw string inputs
        rouge_scores = rouge_evaluator.get_scores(summary, original_text)
        if isinstance(rouge_scores, list):
            rouge_scores = rouge_scores[0]

        # Tokenize for BLEU, METEOR, and compression
        reference_tokens = word_tokenize(original_text)
        candidate_tokens = word_tokenize(summary)

        # BLEU
        bleu_score_val = sentence_bleu([reference_tokens], candidate_tokens)

        # BERTScore
        p, r, f1 = bert_score.score(
            [summary], [original_text], rescale_with_baseline=True, lang='en'
        )

        # METEOR using token lists; returns 0 if no alignment
        meteor_val = meteor([reference_tokens], candidate_tokens)

        # Compression ratio
        original_len = len(reference_tokens)
        summary_len = len(candidate_tokens)
        compression_ratio = summary_len / original_len if original_len > 0 else 0

        # Readability
        readability = textstat.flesch_reading_ease(summary)

        # Aggregate metrics
        row = {
            'model_name': model_name,
            'uuid': uuid,
            'companyid': company_id,
            'companyname': company_name,
            'bleu': bleu_score_val,
            'bert_precision': p.item(),
            'bert_recall': r.item(),
            'bert_f1': f1.item(),
            'meteor': meteor_val,
            'compression_ratio': compression_ratio,
            'readability': readability,
            'type': summarization_type,
        }
        for m, scores in rouge_scores.items():
            row[f'{m}_r'] = scores['r']
            row[f'{m}_p'] = scores['p']
            row[f'{m}_f'] = scores['f']

        results.append(row)

    return pd.DataFrame(results)
