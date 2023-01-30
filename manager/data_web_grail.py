import json
import pickle
import torch

from utils.logger import logger
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    # vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    return vocab


def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    target_ids = torch.stack(batch[2])

    return source_ids, source_mask, target_ids


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids = inputs
        self.is_test = len(self.target_ids)

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        if self.is_test == 0:
            target_ids = None
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
        return source_ids, source_mask, target_ids

    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)

        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(3):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )
        self.vocab = vocab
