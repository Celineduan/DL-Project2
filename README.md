# DL-Project2
Finetuning with LoRA
**Goal**
In this Kaggle competition you are tasked with coming up with a modified BERT architecture with the highest test accuracy on a held-out portion of a text classification dataset called “AGNEWS”, under the constraint that your model has no more than 1 million trainable parameters.
You will start with a specific version of BERT (called RoBERTa). The only type of modification you are allowed to make to BERT is low-rank adaptation [LoRA]. See here or here for an explanation of what LoRA is; the high level idea is that each (frozen) weight matrix in BERT is perturbed by a trainable low-rank matrix, where you can set the rank and the strength of the perturbation. More details are below, and there is also a demo notebook posted on the Kaggle competition site, which uses a popular LLM fine-tuning library called BitsAndBytes.
**Details about LoRA**
Here is a schematic of how LoRA looks like:
There are two main advantages of doing this. First, there are far fewer trainable parameters, so gradient computation is much easier and can be done on much cheaper hardware. The second is we keep the pre-trained weights frozen and only train (adapt) the new weights to the given task, so it is easy to switch out adapters, merge more than one adapter, etc.
The main control knob in LoRA is the rank of the matrix AB (ie the smaller dimension of either A or B), denoted by r. Another control knob is the strength, α, which is a scalar multiplied with AB when added to the original weight. Small values of α are used when we don’t want too much change in the base model weights, and vice versa.
