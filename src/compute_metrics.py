import string
import re
import json
import sys
import os
import logging
from collections import Counter
from rouge import rouge_scorer


logger = logging.getLogger(__name__)


class SplitTokenizer:
    def tokenize(self, s):
        return s.split()


# copy the flowing from Squad v1.1 evaluation
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    # def remove_articles(text):
    #     return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def rouge1_score(prediction, ground_truth, non_en=False):
    if not non_en:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=SplitTokenizer()) 
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, non_en=False):
    if not non_en:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=SplitTokenizer()) 
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def compute_metrics(predictions, references):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, f1, rouge1, rougeL = 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold
        )
        f1 += metric_max_over_ground_truths(
            f1_score, prediction=pred, ground_truths=gold
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold
        )
    exact_match = 100.0 * exact_match / len(references)
    f1 = 100.0 * f1 / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {'exact_match': exact_match, 'f1': f1, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


if __name__ == "__main__":
    with open(sys.argv[1]) as fin:
        examples = [json.loads(l) for l in fin]
        examples = [e for e in examples if not e["Task"].startswith("task1415_")] # remove task1415
    
    if "gpt3" in os.path.basename(sys.argv[1]):
        for example in examples:
            example["Prediction"] = example["gpt3_response"]["choices"][0]["text"].strip().split(".")[0]

    predictions = [e["Prediction"] for e in examples]
    references = [e["Instance"]["output"] for e in examples]
    tasks = []
    for e in examples:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    results = compute_metrics(predictions, references)
    print("all_rougeL", results["rougeL"])
    print("all_EM", results["exact_match"])
    
    task_category = {}
    for task in set(tasks):
        with open(os.path.join("./data/tasks/", task+".json")) as fin:
            task_data = json.load(fin)
            task_category[task] = "_".join(task_data["Categories"][0].lower().split())
    categories = [task_category[e["Task"]] for e in examples] 
    results.update(compute_grouped_metrics(predictions, references, categories))
    category_metrics = [
        ("Textual Entailment", "exact_match"),
        ("Cause Effect Classification", "exact_match"),
        ("Coreference Resolution", "exact_match"),
        ("Dialogue Act Recognition", "exact_match"),
        ("Answerability Classification", "exact_match"),
        ("Word Analogy", "exact_match"),
        ("Overlap Extraction", "rougeL"),
        ("Keyword Tagging", "rougeL"),
        ("Question Rewriting", "rougeL"),
        ("Title Generation", "rougeL"),
        ("Data to Text", "rougeL"),
        ("Grammar Error Correction", "rougeL"),
    ]
    for category, metric in category_metrics:
        category = "_".join(category.lower().split())
        if f"{metric}_for_{category}" in results:
            print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
            
    category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}
    results_by_task = compute_grouped_metrics(predictions, references, tasks)
    for task in sorted(list(set(tasks))):
        category = task_category[task]
        metric = category_metrics[category]
        print(task, results_by_task[f"{metric}_for_{task}"])