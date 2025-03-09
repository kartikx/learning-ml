import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hyperparams import *
import input


class Head(nn.Module):
    """
    A single head of attention.
    Given an input, calculate K, Q, V matrices.
    Apply attention.
    Return output.
    """

    def __init__(self):
        super().__init__()

        self.w_q = nn.Linear(n_embd, dim_q, bias=False)
        self.w_k = nn.Linear(n_embd, dim_k, bias=False)
        self.w_v = nn.Linear(n_embd, dim_v, bias=False)

        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        # x (B, T, n_embd)
        q = self.w_q(x)  # (B, T, dim_q)
        k = self.w_k(x)  # (B, T, dim_k)
        v = self.w_k(x)  # (B, T, dim_v)

        attention = (q @ torch.transpose(k, -2, -1) /
                     math.sqrt(dim_k))  # (B, T, T)

        # mask qk, set upper diagonal to -inf.
        attention = torch.masked_fill(attention, self.tril == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)  # (B, T, T)

        out = attention @ v  # (B, T, dim_v)

        return out


class MultiHead(nn.Module):
    """
    Multiple heads of attention.
    This should have num_heads heads in a sequential layer.
    The forward pass is going to be just applications of all together, followed by a concatenation.
    """

    def __init__(self):
        super().__init__()

        self.heads = [Head() for i in range(num_heads)]

    def forward(self, x):
        # x is from embedding layer (B, T, n_embd)
        # num_heads * (B, T, dim_v)
        head_outs = [head(x) for head in self.heads]

        # convert a list into a tensor.
        # (B, T, num_heads * dim_v) or (B, T, n_embd)
        out = torch.cat(head_outs, axis=-1)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_embd, n_embd * 4),
                                    nn.ReLU(),
                                    nn.Linear(n_embd*4, n_embd))

    def forward(self, x):
        # x is (B, T, n_embd)
        out = self.layers(x)  # (B, T, n_embd)
        return out


class AttentionBlock(nn.Module):
    """
    One block of Attention followed by a Feed Forward Network.
    """

    def __init__(self):
        super().__init__()

        self.multi_head = MultiHead()
        self.linear = FeedForward()
        # TODO residual connection. layer norm.

    def forward(self, x):
        # x is (B, T, n_embd)
        out = self.multi_head(x)
        out = self.linear(out)
        return out


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_table = nn.Embedding(
            input.vocab_size, n_embd)
        self.position_table = nn.Embedding(
            block_size, n_embd)

    # x = {B, T}
    def forward(self, x):
        B, T = x.shape
        token_embedding = self.embedding_table(x)  # (B, T, n_embd)
        positional_embedding = self.position_table(
            torch.arange(0, T))  # (B, T, n_embd)

        return token_embedding + positional_embedding


class GPT2(nn.Module):
    """
    GPT2 model, combines heads and FFNs.
    """

    def __init__(self):
        super().__init__()

        self.embedding = EmbeddingLayer()
        self.blocks = nn.Sequential(*[AttentionBlock()
                                    for _ in range(num_blocks)])
        self.output = nn.Linear(n_embd, input.vocab_size)

    def forward(self, x):
        # x is (B, T)
        out = self.embedding(x)  # (B, T, n_embd)
        out = self.blocks(out)   # (B, T, n_embd)
        out = self.output(out)   # (B, T, vocab_size)
        return out
