# tinygpt (all lowercase)

tinygpt trains a transfomer-based language model on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). It is intended as a teaching resource, so most things are implemented from scratch with minimal abstractions.

`tinystories.py` downloads the dataset, tokenizes it with Eleuther AI's GPT-Neo-125M tokenizer, and saves it in shards of 100M tokens per shard in the `tinystories` directory. There are 474M tokens in the TinyStories sdataset.

`gpt.py` is a utility file that specifies the parts of the model (the actual model configuration in terms of the parts is in `tinygpt.py`). 

`tinygpt.py` is the main file that configures the model, implements the data loader, and trains the model. The model has 64M parameters, and we train it with a batch size of 500k tokens. It takes 20-30 mins to make a pass thru the dataset on a 2x 4090 node, achieving a validation loss of 1.7-ish. It supports training on multiple GPUs (on a single node) for now, but I may remove this functionality in the future after I shrink the model more so that training on a single GPU is practical.

`nb.ipynb` is a notebook that analyzes the training data and generates some completions. 