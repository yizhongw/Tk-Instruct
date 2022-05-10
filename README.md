# Tk-Instruct

- This repo releases our implementation for the Tk-Instruct model in the [Natural Instructions V2 paper](https://arxiv.org/abs/2204.07705).
- Tk-Instruct is a preliminary attempt towards general-purpose AI that can solve NLP tasks by following in-context instructions (plain language task definitions or k-shot examples).
- It is built based on the pretrained [T5 model](https://arxiv.org/abs/1910.10683), and finetuned on our [data](https://github.com/allenai/natural-instructions).
- We are going to release our demo soon. Stay tuned!
<!-- - You can play with this model via our online [demo](https://instructions.apps.allenai.org/demo)! -->

## Requirements

Our main experiments and analysis are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```bash
pip install -r requirements.txt
```

Note: after the main exploration with 3B model, we train our 11B model on TPUs using the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer).

## Data

Our models are trained and evaluated on [Natural Instructions V2](https://github.com/allenai/natural-instructions), which can be cloned by running:

```bash
git clone git@github.com:allenai/natural-instructions.git data
```

Since Natural Instructions V2 didn't provide an official split for the development set, in order to do evaluation during training time, you can mannualy create a `dev_tasks.txt` in the `data/splits/default` folder. We found it unclear what should be a meaningful validation set, under such cross-task generalization setting. You can use a part of the training tasks for validation, or you can set apart tasks in some categories for validation.

If you want to use the T5 code [here](https://github.com/google-research/text-to-text-transfer-transformer), you can convert the data into text2text format with [`scripts/convert_data_to_s2s.sh`](scripts/convert_data_to_s2s.sh).

## Training

A sample script for training the Tk-Instruct 3B model in our paper can be found at [`scripts/train_tk_instruct.sh`](scripts/train_tk_instruct.sh). You can run it as follows:

```bash
./scripts/train_tk_instruct.sh
```

However, if you are familiar with [Beaker](https://beaker.org/), you can refer to the [`beaker_configs/default_experiment.yaml`](beaker_configs/default_experiment.yaml) for a sample experiment config, and modifying [`src/create_exps.py`](src/create_exps.py) to easily starts a set of experiments by running:

```bash
python src/create_exps.py
```

## Released Checkpoints

Our 3B and 11B model checkpoints are accessible via the [Hugging Face Hub](https://huggingface.co/models?search=tk-instruct-). You can load them easily using the [Transformers](https://github.com/huggingface/transformers) library:

```python
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

>>> tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
>>> model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def")

>>> input_ids = tokenizer.encode(
        "Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
        return_tensors="pt"
    )
>>> output = model.generate(input_ids, max_length=10)
>>> output = tokenizer.decode(output[0], skip_special_tokens=True)
```

The model should generate `'Indian Rupee'` as the output.

## Evaluation

The following script evaluates our 3B Tk-Instruct model that uses `task definition + 2 positive examples` as instructions:

```bash
./scripts/eval_tk_instruct.sh
```

This should give you a ROUGE-L score of ~54.0, as is reported in the Table 3 of our [paper](https://arxiv.org/pdf/2204.07705.pdf).

You can also try other models under different encodings. You can control whether to include definition / explanation, or the number of pos/neg examples, by specifying the arguments in [`src/run_s2s.py`](src/run_s2s.py).

The numbers for heuristic baselines and GPT3 can be reproduced by using the following scripts:

```bash
./scripts/run_heuristics.sh
./scripts/run_gpt3.sh
```

## Model Predictions

We are going to release the predictions of our models and baseline methods soon!

## Citation

```bib
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
```
