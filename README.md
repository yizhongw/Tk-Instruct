# Tk-Instruct

- This repo releases our implementation for the Tk-Instruct model in the [Natural Instructions V2 paper](https://arxiv.org/abs/2204.07705).
- Tk-Instruct is a preliminary attempt towards general-purpose AI that can solve new tasks by following instructions. It is built based on the pretrained [T5 model](https://arxiv.org/abs/1910.10683).
- You can play with this model via our online [demo](https://instructions.apps.allenai.org/demo)!

## Requirements

Our experiments are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```bash
pip install -r requirements.txt
```

## Data

## Training

A sample script for training the Tk-Instruct 3B model in our paper can be found at [`scripts/train_tk_instruct.sh`](scripts/train_tk_instruct.sh). You can run it as follows:
```bash
./scripts/train_tk_instruct.sh
```

However, if you are familiar with [Beaker](https://beaker.org/), you can refer to the [`beaker_configs/default_experiment.yaml`](beaker_configs/default_experiment.yaml) for a sample experiment config, and modifying [`src/create_exps.py`](src/create_exps.py) to easily starts a set of experiments.

## Evaluation

## Predictions and Results

## Checkpoints

Our 3B and 11B model checkpoints are accessible via [Huggingface Hub](https://huggingface.co/models?search=tk-instruct-).
