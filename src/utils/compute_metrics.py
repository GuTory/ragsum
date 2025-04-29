import pandas as pd
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import textstat
import bert_score

def compute_metrics(originals: list, summaries: list, model_name, type='baseline'):
    results = []
    rouge_evaluator = Rouge()

    for original_text, summary in zip(originals, summaries):
        # Basic metadata placeholder — adjust as needed if available
        uuid = company_id = company_name = None  # Replace with actual data if available

        # ROUGE
        rouge_scores = rouge_evaluator.get_scores(summary, original_text)
        if isinstance(rouge_scores, list):
            rouge_scores = rouge_scores[0]

        # BLEU
        reference_tokens = original_text.split()
        candidate_tokens = summary.split()
        bleu_score_val = sentence_bleu([reference_tokens], candidate_tokens)

        # BERTScore
        P, R, F1 = bert_score.score(
            [summary], [original_text], rescale_with_baseline=True, lang='en'
        )

        # METEOR
        meteor = meteor_score([original_text], summary)

        # Compression
        original_len = len(reference_tokens)
        summary_len = len(candidate_tokens)
        compression_ratio = summary_len / original_len if original_len > 0 else 0

        # Readability
        readability = textstat.flesch_reading_ease(summary)

        # Result aggregation
        row_result = {
            'model_name': model_name,
            'uuid': uuid,
            'companyid': company_id,
            'companyname': company_name,
            'bleu': bleu_score_val,
            'bert_precision': P.item(),
            'bert_recall': R.item(),
            'bert_f1': F1.item(),
            'meteor': meteor,
            'compression_ratio': compression_ratio,
            'readability': readability
        }

        for metric, scores in rouge_scores.items():
            row_result[f'{metric}_r'] = scores['r']
            row_result[f'{metric}_p'] = scores['p']
            row_result[f'{metric}_f'] = scores['f']

        results.append(row_result)

    return pd.DataFrame(results)
