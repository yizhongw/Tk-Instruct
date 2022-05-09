import os
import json
import glob
from compute_metrics import compute_metrics, compute_grouped_metrics
from transformers import HfArgumentParser, GPT2TokenizerFast
from run_s2s import DataTrainingArguments
from datasets import load_dataset
from ni_collator import DataCollatorForNI
from dataclasses import dataclass, field
from nltk import sent_tokenize


@dataclass
class BaselineArguments(DataTrainingArguments):
    method: str = field(
        default="copy_demo", metadata={"help": "The baseline method, including copy_demo or copy_input."}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((BaselineArguments,))
    args, = parser.parse_args_into_dataclasses()
    raw_datasets = load_dataset(
        "src/ni_dataset.py",
        data_dir=args.data_dir, 
        task_dir=args.task_dir, 
        max_num_instances_per_task=args.max_num_instances_per_task,
        max_num_instances_per_eval_task=args.max_num_instances_per_eval_task
    )

    examples = raw_datasets["test"]
    examples = [e for e in examples if not e["Task"].startswith("task1415_")] # remove task1415
    tasks = []
    for e in examples:
        if e["Task"] == "task121_atomic_question_rewriting":
            e["Task"] = "task121_zest_question_rewriting"
        tasks.append(e["Task"])

    predictions, references = [], []
    for example in examples:
        if example["Task"].startswith("task1415_"):
            # print(example["Task"])
            continue
        if args.method == "copy_demo":
            predictions.append(example["Positive Examples"][0]["output"])
        elif args.method == "copy_input":
            # first_sent = sent_tokenize(example["Instance"]["input"])[0]
            # predictions.append(first_sent)
            predictions.append(example["Instance"]["input"])
        references.append(example["Instance"]["output"])

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
    tasks = [e["Task"] for e in examples] 
    results_by_task = compute_grouped_metrics(predictions, references, tasks)
    for task in sorted(list(set(tasks))):
        category = task_category[task]
        metric = category_metrics[category]
        print(task, results_by_task[f"{metric}_for_{task}"])