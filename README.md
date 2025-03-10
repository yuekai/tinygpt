# tinygpt (all lowercase)

Pretrain a transfomer model on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). The architecture is similar to that of [GPT-2](https://huggingface.co/openai-community/gpt2). It is intended as a teaching resource, so most things are implemented from scratch with minimal abstractions.

`tinystories.py` downloads the dataset, tokenizes it with Eleuther AI's GPT-Neo-125M tokenizer, and saves it in shards of 100M tokens per shard in the `tinystories` directory. There are 474M tokens in the TinyStories sdataset.

`gpt.py` is a utility file that specifies a generic transformer language model (the actual model configuration is in `tinygpt.py`). 

`tinygpt.py` is the main file that configures the model, implements the data loader, and trains the model. The model has 61M parameters, and we train it with batches of 500k tokens. It takes 30-ish mins to make a pass thru the dataset on a 2x 4090 node, so a 10 epoch training run takes 5-ish hrs (achieving a validation loss of 1.32). The main difference between this and Andrej Karpathy's implementation is this implementation enforces attention masking to prevent the attention mechanism from attending to tokens in other stories (in the same batch).

1. The file supports training on multiple GPUs (on a single node) for now, but I may remove this functionality in the future after I shrink the model more so that training on a single GPU is practical.
2. `torchinfo` summarizes the model as having 86M (trainable) parameters, but the weights of the final linear layer and the token embedding layer are shared, so the actual number of parameters is 61M. You can check this by checking `model.transformer.wte.weight = model.lm_head.weight`.
3. training hyperparameters are those of [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M)

`nb.ipynb` is a notebook that analyzes the training data and generates some text. Here is a sample:

```
<|endoftext|>Mum and Dad wanted to go out to the factory with their things. 

Mum said it was too early to go there and the factory wasn't too deep.

Mum and Dad tried to stay still but could not go ahead.

Suddenly, Mum said it had to be safe and play. She was worried about what she should do about it. 

When Dad saw what Mum and Dad doing. He ran over to them and helped put out the things they were supposed to.

Then, he smiled and said they could go.

Mum and Dad were surprised to see an old man who was deaf at the back of the factory. Dad laughed and said he was a little too. 

Mum and Dad said it was ok too. Even though they were happy, they were relieved that they were so excited. 

The End.<|endoftext|>Once there was a little girl called Sarah. She was three years old and
```