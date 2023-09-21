import argparse
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
import math

def predict_next_token_probabilities(model, input_ids, alpha, intermediate_layer, temperature):
    output = model(input_ids, use_cache=True, output_hidden_states=True)

    next_token_logits = output.logits.squeeze(0)[-1]
    log_probabilities = F.log_softmax(next_token_logits, dim=0)

    intermediate_output = output.hidden_states[intermediate_layer].squeeze(0)[-1]
    intermediate_logits = model.lm_head(model.transformer.ln_f(intermediate_output))
    intermediate_log_probs = F.log_softmax(intermediate_logits / temperature, dim=0)

    max_log_prob = torch.max(log_probabilities)
    log_alpha = torch.tensor(math.log(alpha))
    
    probs = []
    for i in range(len(log_probabilities)):
        if log_probabilities[i] >= log_alpha + max_log_prob:
            score = log_probabilities[i] - intermediate_log_probs[i]
            probs.append((torch.tensor(i), -score.item()))

    return probs

def contrastive_beam_search(model, initial_sequence, beam_width, sequence_length, alpha, intermediate_layer, temperature):
    sequences = [(0, initial_sequence)]  # (score, sequence)
    for i in range(sequence_length - len(initial_sequence) + 1):  # Corrected the range
        all_possible_next_sequences = []
        for score, sequence in sequences:
            next_token_probabilities = predict_next_token_probabilities(model, sequence, alpha, intermediate_layer, temperature)
            for next_token, next_token_log_prob in next_token_probabilities:
                new_score = score + next_token_log_prob
                new_sequence = torch.cat((sequence.squeeze(0), next_token.unsqueeze(0)), dim=0).unsqueeze(0)
                all_possible_next_sequences.append((new_score, new_sequence))
        # Keep top beam_width sequences based on their score
        ordered = sorted(all_possible_next_sequences, key=lambda tup: tup[0])
        sequences = ordered[:beam_width]
    return sequences


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-l', '--max_length', type=int, default=100)
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-i', '--intermediate_layer', type=int, required=True)
    parser.add_argument('-t', '--temperature', type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    encoded_input = tokenizer.encode(args.prompt, return_tensors='pt')

    print('Contrastive:')
    bso = contrastive_beam_search(
        model,
        encoded_input,
        beam_width=5,
        sequence_length=args.max_length,
        alpha=args.alpha,
        intermediate_layer=args.intermediate_layer,
        temperature=args.temperature
    )
    _, sequence = bso[0]
    print(tokenizer.decode(sequence.squeeze(0), skip_special_tokens=True))
    print()

    output = model.generate(encoded_input, max_length=args.max_length, num_beams=5)
    print('Beam search:')
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()
