import torch
import input
from model import GPT2
from train import train_model, generate_output
from hyperparams import block_size

"""
Process:
2. Code out each layer.
3. Test output (dimensionality check)
4. Train model.
"""


def main():
    input.parse_input("./input.txt")
    model = GPT2()
    train_model(model)
    # ? Why zeros in particular?
    input_text = torch.zeros((1, block_size), dtype=torch.long)
    output = generate_output(model, input_text)
    print("Generated: \n", output)


if __name__ == '__main__':
    main()
