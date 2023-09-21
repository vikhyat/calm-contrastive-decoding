# Confident Adaptive Language Modeling 🫶 Contrastive Decoding

The [Contrastive Decoding](https://arxiv.org/abs/2210.15097) paper describes a technique to product higher quality text from language models by using the difference between predictions from a large LM (called the expert, e.g. OPT-13B), and a small LM (called the amateur, e.g. OPT-125M). This repository contains the code for an experiment to see whether we can use early exit, as described in the [Confident Adaptive Language Modeling](https://arxiv.org/abs/2207.07061) paper, in lieu of the amateur model to reduce the computational cost of contrastive decoding.

## Experimental Setup

We'll use the [GPT-2 XL](https://huggingface.co/gpt2-xl) model since it was also used in the Contrastive Decoding paper. GPT-2 has a [pre-LayerNorm](https://arxiv.org/abs/2002.04745) decoder-only transformer architecture with tied weights in the token embedding matrix and the linear layer of the language modeling head.

### Early Exit

The tied weights introduces a complication because the CALM paper uses an 8 layer T5 1.1 model for their experiments which does not share input and output embeddings. They do, however, share output embeddings for the intermediate layer softmax predictions with the top one. This turns out to work well for GPT-2 as well, meaning we don't need to do any additional training to get the early exit logits. We just need to apply the final layer norm to the intermediate layer logits and then apply the same language modeling head.

![Early Exit Architecture](assets/early_exit_architecture.jpg)

Here's an example of how logit probabilities evolve as we move through layers of the model for a sample input sequence.

![Early Exit Probabilities](assets/early_exit_probabilities.jpg)

### Contrastive Decoding

The key idea behind the contrastive decoding technique is to push model predictions away from predictions made by the smaller language model. However this introduces false positives and negatives for tokens that have low probability in the expert model. To address this, we filter out tokens that have a probability lower than α times the maximum token probability at that position. The paper uses α = 0.1.

Once low probability tokens are filtered out, tokens are scored using the difference between log probabilities from the expert and amateur models. The paper also finds that a temperature of 0.5 on the amateur model provides the best results. Beam search (with a beam size of 5) is used to find the best sequence of tokens.

## Results

(coming soon)