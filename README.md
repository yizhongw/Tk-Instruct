# Tk-Instruct

- This repo releases our implementation for the Tk-Instruct model in the [Natural Instructions V2 paper](https://arxiv.org/abs/2204.07705).
- Tk-Instruct is a preliminary attempt towards general-purpose AI that can solve new tasks by following instructions.
- You can play with this model via our online [demo](https://instructions.apps.allenai.org/demo)!

## Requirements

Our experiments are conducted on the following environment:

- CUDA (11.3)
- cuDNN (8.2.0.53)
- Pytorch (1.10.0)
- Transformers (4.17.0)
- DeepSpeed

You can refer to the [Dockerfile](Dockerfile) for setting up the environment and install the required python libraries by running

```script
pip install -r requirements.txt
```

## Data

## Training

## Evaluation

## Predictions and Results

## Checkpoints

Our 3B and 11B model checkpoints are accessible via [Huggingface Hub](https://huggingface.co/models?search=tk-instruct-).
