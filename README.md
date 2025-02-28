# tinygpt (all lowercase)

Pretrain a transfomer model on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). The architecture is similar to that of [GPT-2](https://huggingface.co/openai-community/gpt2). It is intended as a teaching resource, so most things are implemented from scratch with minimal abstractions.

`tinystories.py` downloads the dataset, tokenizes it with Eleuther AI's GPT-Neo-125M tokenizer, and saves it in shards of 100M tokens per shard in the `tinystories` directory. There are 474M tokens in the TinyStories sdataset.

`gpt.py` is a utility file that specifies the parts of the model (the actual model configuration in terms of the parts is in `tinygpt.py`). 

`tinygpt.py` is the main file that configures the model, implements the data loader, and trains the model. The model has 61M parameters, and we train it with batches of 500k tokens. It takes 25-30 mins to make a pass thru the dataset on a 2x 4090 node, so a 10 epoch training run takes 4-5 hrs (achieving a validation loss of 1.32). 

1. The file supports training on multiple GPUs (on a single node) for now, but I may remove this functionality in the future after I shrink the model more so that training on a single GPU is practical.
2. `torchinfo` summarizes the model as having 86M (trainable) parameters, but the weights of the final linear layer and the token embedding layer are shared, so the actual number of parameters is 61M. You can check this by checking `model.transformer.wte.weight = model.lm_head.weight`.
3. training hyperparameters are those of [TinyStories-33M](https://huggingface.co/roneneldan/TinyStories-33M)

`nb.ipynb` is a notebook that analyzes the training data and generates some text. Here is a sample:

```
<|endoftext|>Once upon a time, there was a chubby bunny named Bongo. Bongo loved to play all day long. One day, Bongo wanted to go outside and play, but it had a tug--ri ran out of exhaustion. Bongo didn't know what to do and he was so grumpy! 

Bongo decided to ask his friend, a bear named Ben, for help. Ben was so happy to help Bongo, but Peppa told him it was too late. Bongo had won and his digging. Bongo was so happy that he tried to help his friend. 

Together, they worked both around the meadow, and Bongo was so proud! After many days, Bongo and Ben became best friends. Bongo was so happy that he forgot all about the scary size of Jack's rattle. The moral of the story is that right friends can help you become a better team.<|endoftext|>Once there was a little girl called Kayla
```

__ToDo:__
- [x] Implement grouped query attention (GQA) to reduce model size.
- [ ] Implement tokenizer from scratch (instead of relying on Eleuther AI's GPT-Neo-125M tokenizer).
- [ ] Implement rotary position embeddings (instead of absolute position embeddings). This (together with GQA) brings the architecture (mostly) in line with the latest open weight LLMs.
- [ ] Reduce the vocabulary from 50k-ish to 5k-ish. I suspect we can get away with a smaller vocabulary because the language in TinyStores is so simple. 