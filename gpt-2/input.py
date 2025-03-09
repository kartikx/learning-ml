from hyperparams import block_size, batch_size
import torch
global char_to_index, index_to_char, vocab_size, data, train_data, test_data, vocab


def encode(text: str):
    return [char_to_index[ch] for ch in text]


def decode(indices):
    return ''.join(index_to_char[index] for index in indices)


def get_batch(split: str):
    """
    Return a batch of samples, depending on the split.
    """

    dataset = train_data if split == "train" else test_data

    # get a random index in range 0, end - block_size
    start_indices = torch.randint(
        0, len(dataset) - block_size, (batch_size, ))

    # for each start_indice i need to build out a sequence.
    x = torch.stack([dataset[index:index+block_size]
                     for index in start_indices])

    y = torch.stack([dataset[index+1:index+block_size+1]
                     for index in start_indices])
    return x, y


def parse_input(file_path: str):
    global char_to_index, index_to_char, test_data, train_data, data, vocab_size, vocab

    with open(file_path, 'r') as f:
        text = f.read()
        vocab = sorted(list(set(text)))

        vocab_size = len(vocab)
        char_to_index = {
            vocab[index]: index for index in range(vocab_size)}
        index_to_char = {index: vocab[index]
                         for index in range(vocab_size)}

        data = torch.tensor(encode(text), dtype=torch.long)

        train_test_split_index = int(0.9 * len(data))
        train_data = data[:train_test_split_index]
        test_data = data[train_test_split_index:]
