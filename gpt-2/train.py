import torch
import torch.optim as optim
import torch.nn.functional as F
from model import GPT2
from input import get_batch, decode
from hyperparams import *


def get_loss(output, target):
    # output -> (B, T, vocab_size), target -> (B, T)
    B, T, V = output.shape
    output = output.view(B * T, V)
    target = target.view(B * T,)
    loss = F.cross_entropy(output, target)
    return loss


def train_model(model):
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        model.zero_grad(set_to_none=True)
        sequence, labels = get_batch('train')
        output = model(sequence)  # (B, T, vocab_size)
        loss = get_loss(output, labels)
        loss.backward()
        optimizer.step()


def generate_output(model, input):
    # input (B, T)

    # TODO, take an input sequence and generate a certain amount of outputs.
    for iter in range(num_decode_steps):
        # (B, T, vocab_size)
        input_curr = input[:, -block_size:]
        output_logits = model(input_curr)  # (B, T, vocab_size)
        output_probs = F.softmax(output_logits, dim=-1)  # (B, T, vocab_size)
        # only keep the last token's logits
        output_probs = output_probs[:, -1, :]  # (B, 1, vocab_size)
        output_probs = output_probs.view(output_probs.shape[0], -1)

        output_probs = torch.nan_to_num(output_probs, nan=1e-8)  # Replace NaNs
        # Ensure no values < 0
        output_probs = torch.clamp(output_probs, min=1e-8)

        next_char = torch.multinomial(output_probs, 1)  # (B, 1)
        input = torch.cat([input, next_char], dim=1)

    input_list = input.tolist()[0][block_size:]
    return decode(input_list)
