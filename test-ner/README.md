---
metrics:
- precision
- recall
- f1
- accuracy

model-index:
- name: un-ner
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# un-ner

This model was trained from scratch on an unkown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0527
- Precision: 0.8046
- Recall: 0.8493
- F1: 0.8263
- Accuracy: 0.9844

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| No log        | 1.0   | 229  | 0.0609          | 0.7480    | 0.7858 | 0.7664 | 0.9794   |
| No log        | 2.0   | 458  | 0.0522          | 0.7923    | 0.8452 | 0.8179 | 0.9837   |
| 0.1066        | 3.0   | 687  | 0.0527          | 0.8046    | 0.8493 | 0.8263 | 0.9844   |


### Framework versions

- Transformers 4.6.1
- Pytorch 1.8.1
- Datasets 1.6.2
- Tokenizers 0.10.2
