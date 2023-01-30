import json
import pickle
import torch
from typing import Dict

from utils.logger import logger
from utils.misc import invert_dict
from conf.data_config import DATA_MAP, BaseData


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    vocab['function_token_to_idx'] = invert_dict(vocab['function_token_to_idx'])
    return vocab


def collate(batch):
    name2data = {}
    for data_name2data in batch:
        for name, data in data_name2data.items():
            name2data.setdefault(name, []).append(data)
    for name, data in list(name2data.items()):
        if name == BaseData.answers.value:
            name2data[name] = torch.cat(data)
        else:
            name2data[name] = torch.stack(data)
    return name2data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, question_pt, model_strategy):
        self.model_strategy = model_strategy
        with open(question_pt, 'rb') as f:
            self.data = self._get_data(pickle.load(f))
        self.is_test = len(self.data[BaseData.answers.value]) == 0
        self.length = len(self.data[BaseData.source_ids.value])

    def __getitem__(self, index):
        data = {}
        for data_name, data_value in self.data.items():
            if data_name == BaseData.answers.value:
                data[data_name] = torch.LongTensor([data_value[index]])
            else:
                data[data_name] = torch.LongTensor(data_value[index])
        return data

    def _get_data(self, all_data: Dict):
        data_type = DATA_MAP.get(self.model_strategy, BaseData)
        data = {}
        for data_name in data_type:
            data[data_name.value] = all_data[data_name.name]
        return data

    def __len__(self):
        return self.length


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, model_strategy=None, training=False):
        vocab = load_vocab(vocab_json)
        if training:
            logger.info('#vocab of answer: %d' % (len(vocab['answer_token_to_idx'])))
        dataset = Dataset(question_pt, model_strategy)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )
        self.vocab = vocab
