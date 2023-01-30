import os
import json
import shutil
import pickle
import numpy as np
import torch
import random
from tqdm import tqdm
from pathlib import Path
from itertools import chain
from transformers import *

from utils.logger import logger
from interfaces.preprocess import Preprocess
from conf.setting import FunctionInputReplaceStrategy, BartSetting, BartModelStrategy
from conf.config import (
    ADD_TRAIN_ANSWER,
    DELIMITER,
    INIT_FUNC_TOKENS,
    FuncTokensType,
    ProgramToken,
    IS_FUNC_TOKENIZER,
    IS_ADD_FUNC_TOKENIZER,
    IS_NEW_DATA,
    NEW_DATASET_NAME,
    FILTER_NEW_DATA_NUM,
    IS_FILTER_NEW_DATA
)

new_tokens = [i.value for i in ProgramToken]
if ADD_TRAIN_ANSWER:
    new_tokens.append(DELIMITER)


class BartPreprocess(Preprocess):

    def __init__(
            self,
            input_dir,
            model_path,
            work_dir,
            conf=BartSetting,
            is_new_data=IS_NEW_DATA,
            new_dataset_name=NEW_DATASET_NAME,
            filter_new_data_num=FILTER_NEW_DATA_NUM,
            is_filter_new_data=IS_FILTER_NEW_DATA
    ):
        self.input_dir = input_dir
        self.model_path = model_path
        self.work_dir = work_dir
        self.conf = conf

        self.is_new_data = is_new_data
        self.new_dataset_name = new_dataset_name
        self.filter_new_data_num = filter_new_data_num
        self.is_filter_new_data = is_filter_new_data

        # self.tree_encoder = SentenceDependencyTreeEncoder()
        self.tree_encoder = None

    def do(self):
        input_dir_obj = Path(self.input_dir)
        name2dataset = {
            'train': self._load_json_file(input_dir_obj.joinpath('train.json')),
            'val': self._load_json_file(input_dir_obj.joinpath('val.json')),
            'test': self._load_json_file(input_dir_obj.joinpath('test.json'))
        }
        vocab = {
            'answer_token_to_idx': {},
            'function_token_to_idx': INIT_FUNC_TOKENS,
            # 'function_token_to_idx': {},
        }
        logger.info('Build kb vocabulary')
        max_func = 0
        for name, dataset in list(name2dataset.items()):
            if self.is_new_data and name in self.new_dataset_name:
                dataset = self._filter_new_data(dataset)
                name2dataset[name] = dataset
                dateset_max_func = self._update_base_resource_by_new_data(vocab, dataset)
            else:
                dateset_max_func = self._update_base_resource(vocab, dataset)
            max_func = max(max_func, dateset_max_func)
        all_func = list(vocab['function_token_to_idx'].keys())
        logger.info(' >>> function max length {}'.format(max_func))
        logger.info(' >>> function num {}, all function {}'.format(len(all_func), all_func))

        self._save_vocab(vocab)
        model, tokenizer = self._load_and_update_model(all_func)

        for name, dataset in name2dataset.items():
            logger.info('Encode {} set'.format(name))
            outputs = self.encode_dataset(dataset, vocab, tokenizer, max_func, name, name == 'test')
            logger.info('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
            with open(os.path.join(self.work_dir, '{}.pt'.format(name)), 'wb') as f:
                for data_name, data_value in outputs.items():
                    logger.info(f'{data_name} >>> {data_value.shape}')
                pickle.dump(outputs, f)
        shutil.copy(Path(self.input_dir).joinpath('kb.json'), Path(self.work_dir).joinpath('kb.json'))

    def encode_dataset(self, dataset, vocab, tokenizer, max_func, name, is_test=False):
        if self.is_new_data and name in self.new_dataset_name:
            questions, programs, choices, answers, functions, function_masks, inputs, max_seq_length, target_random_seqs = self._preprocess_data_by_new_data(
                dataset, vocab, tokenizer, max_func, is_test
            )
        else:
            questions, programs, choices, answers, functions, function_masks, inputs, max_seq_length, target_random_seqs = self._preprocess_data(
                dataset, vocab, tokenizer, max_func, is_test
            )
        # input
        input_ids = tokenizer.batch_encode_plus(
            questions, max_length=max_seq_length, pad_to_max_length=True, truncation=True
        )
        source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
        source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)

        # 树模型向量
        if self.conf.model_strategy == BartModelStrategy.TREE_INFO.value:
            tree_data = self.tree_encoder.batch_encode(questions)
        else:
            tree_data = torch.Tensor([])

        if is_test:
            target_ids = np.array([], dtype=np.int32)
            tar_mask_ids = np.array([], dtype=np.int32)
            function_ids = np.array([], dtype=np.int32)
            function_masks = np.array([], dtype=np.int32)
            inputs_ids = np.array([], dtype=np.int32)
            input_masks = np.array([], dtype=np.int32)
        else:
            if IS_FUNC_TOKENIZER:
                encoded_funcs = tokenizer(functions, padding=True)
                max_func_str_len = len(encoded_funcs['input_ids'][0])
                assert max_func_str_len == len(encoded_funcs['input_ids'][-1])
                logger.info('max_func_str_len >>> {}'.format(max_func_str_len))
                functions = tokenizer.batch_encode_plus(
                    functions, max_length=max_func_str_len, pad_to_max_length=True, truncation=True
                )
                function_ids = np.array(functions['input_ids'], dtype=np.int32)
                function_masks = np.array(functions['attention_mask'], dtype=np.int32)
            else:
                function_ids = np.array(functions, dtype=np.int32)
                function_masks = np.array(function_masks, dtype=np.int32)

            encoded_inp = tokenizer(inputs, padding=True)
            max_input_str_len = len(encoded_inp['input_ids'][0])
            assert max_input_str_len == len(encoded_inp['input_ids'][-1])
            logger.info("max_input_str_len >>> {}".format(max_input_str_len))
            # 目标label
            target_ids = tokenizer.batch_encode_plus(
                programs, max_length=max_seq_length, pad_to_max_length=True, truncation=True
            )
            target_ids = np.array(target_ids['input_ids'], dtype=np.int32)
            input_length = max([max_seq_length, max_input_str_len])
            logger.info(f'input_length >>> {input_length}')
            inputs = tokenizer.batch_encode_plus(
                inputs, max_length=input_length, pad_to_max_length=True, truncation=True
            )
            inputs_ids = np.array(inputs['input_ids'], dtype=np.int32)
            input_masks = np.array(inputs['attention_mask'], dtype=np.int32)

            # 对比学习目标负样本label
            tar_mask_ids = tokenizer.batch_encode_plus(
                target_random_seqs, max_length=max_seq_length, pad_to_max_length=True, truncation=True
            )
            tar_mask_ids = np.array(tar_mask_ids['input_ids'], dtype=np.int32)

        choices = np.array(choices, dtype=np.int32)
        answers = np.array(answers, dtype=np.int32)
        all_data = {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'answers': answers,
            'choices': choices,
            # func
            'function_ids': function_ids,
            'function_masks': function_masks,
            'inputs_ids': inputs_ids,
            'input_masks': input_masks,
            # cl
            'tar_mask_ids': tar_mask_ids,
            # tree
            'tree_data': tree_data
        }
        return all_data

    @staticmethod
    def _update_base_resource(vocab, dataset):
        logger.info('Load questions')
        max_func = 0
        for question in dataset:
            for a in question['choices']:
                if not a in vocab['answer_token_to_idx']:
                    vocab['answer_token_to_idx'][a] = len(vocab['answer_token_to_idx'])
            programs = question.get('program', [])
            if len(programs) > max_func:
                max_func = len(programs)
            for program in programs:
                if not program['function'] in vocab['function_token_to_idx']:
                    vocab['function_token_to_idx'][program['function']] = len(vocab['function_token_to_idx'])
        max_func += len(FuncTokensType)  # 长度加上s, e, p
        return max_func

    @staticmethod
    def _update_base_resource_by_new_data(vocab, dataset):
        logger.info('Load new data questions')
        max_func = 0
        for question_item in dataset:
            for choice in question_item['choices']:
                if choice not in vocab['answer_token_to_idx']:
                    vocab['answer_token_to_idx'][choice] = len(vocab['answer_token_to_idx'])
            functions = question_item.get('function', [])
            max_func = max(max_func, len(functions))
            for function in functions:
                if function not in vocab['function_token_to_idx']:
                    vocab['function_token_to_idx'][function] = len(vocab['function_token_to_idx'])
        max_func += len(FuncTokensType)  # 长度加上s, e, p
        return max_func

    def _load_and_update_model(self, all_func):
        model = BartForConditionalGeneration.from_pretrained(self.model_path)
        tokenizer = BartTokenizer.from_pretrained(self.model_path)
        for token in new_tokens:
            tokenizer.add_tokens(token, special_tokens=True)
        if IS_ADD_FUNC_TOKENIZER:
            for func in all_func:
                tokenizer.add_tokens(func, special_tokens=True)
        if len(new_tokens) > 0 or (len(all_func) > 0 and IS_ADD_FUNC_TOKENIZER):
            model.resize_token_embeddings(len(tokenizer))
        model.save_pretrained(os.path.join(self.work_dir, 'model'))
        tokenizer.save_pretrained(os.path.join(self.work_dir, "tokenizer"))
        return model, tokenizer

    def _preprocess_data(self, dataset, vocab, tokenizer, max_func, is_test):
        questions, programs, choices, answers, functions, function_masks, inputs = [], [], [], [], [], [], []
        target_random_seqs = []  # 对比学习负样本
        max_func_len, max_input_len = 0, 0
        all_funcs = []
        for item in tqdm(dataset):
            question = item['question']
            questions.append(question)
            _ = [vocab['answer_token_to_idx'][w] for w in item['choices']]
            choices.append(_)
            if not is_test:
                program = item['program']
                if IS_FUNC_TOKENIZER:
                    function = [func_info['function'] for func_info in program]
                    function = ' '.join(function)
                else:
                    # 不分类时，使用序列生成，可以不给function编码和pad
                    function = [vocab['function_token_to_idx'][FuncTokensType.FUNC_START.value]] + \
                               [vocab['function_token_to_idx'][func_info['function']] for func_info in program] + \
                               [vocab['function_token_to_idx'][FuncTokensType.FUNC_END.value]]
                    pad_len = (max_func - len(function))
                    function = function + [vocab['function_token_to_idx'][FuncTokensType.FUNC_PAD.value]] * pad_len
                    function_mask = [1] * len(function) + [
                        vocab['function_token_to_idx'][FuncTokensType.FUNC_PAD.value]] * pad_len
                    if len(function) != len(function_mask):
                        print("len function_mask", len(function_mask))
                        print("len fuction", len(function))
                    function_masks.append(function_mask)
                    # merge_func_str = ' '.join(function)
                functions.append(function)

                if len(function) > max_func_len:
                    max_func_len = len(function)
                all_funcs.append(function)

                program, pro_sequence, pro_functions, pro_inputs = self._get_program_seq(program)

                input_str = self.get_joint_inputs(pro_inputs)
                inputs.append(input_str)
                if ADD_TRAIN_ANSWER:
                    program += item['answer']
                programs.append(program)
                answers.append(vocab['answer_token_to_idx'].get(item['answer']))

                # 构造cl负样本数据
                if self.conf.replace_method == FunctionInputReplaceStrategy.FUNC.value:
                    ...
                elif self.conf.replace_method == FunctionInputReplaceStrategy.INPUT.value:
                    ...
                elif self.conf.replace_method == FunctionInputReplaceStrategy.OR.value:
                    target_random = [
                        ps if np.random.random() > self.conf.replace_value else np.random.choice(pro_sequence)
                        for ps in pro_sequence]
                    target_random_seq = f' {ProgramToken.FUNC.value} '.join(target_random)
                    target_random_seqs.append(target_random_seq)
                else:
                    raise ValueError(f'{self.conf.replace_method} Not defined!')
        sequences = questions + programs  # 163
        logger.info('tokenizing')
        encoded_inputs = tokenizer(sequences, padding=True)

        logger.info('tokenize ended.')
        logger.info(encoded_inputs.keys())
        logger.info(encoded_inputs['input_ids'][0])
        logger.info(tokenizer.decode(encoded_inputs['input_ids'][0]))
        logger.info(tokenizer.decode(encoded_inputs['input_ids'][-1]))
        max_seq_length = len(encoded_inputs['input_ids'][0])
        assert max_seq_length == len(encoded_inputs['input_ids'][-1])
        logger.info("questions + programs max length: {}".format(max_seq_length))
        return questions, programs, choices, answers, functions, function_masks, inputs, max_seq_length, target_random_seqs

    def _preprocess_data_by_new_data(self, dataset, vocab, tokenizer, max_func, is_test):
        logger.info('Start processing new data')
        questions, programs, choices, answers, functions, function_masks, inputs = [], [], [], [], [], [], []
        target_random_seqs = []  # 对比学习负样本
        for item in tqdm(dataset):
            questions.append(item['question'])
            choices.append([vocab['answer_token_to_idx'][w] for w in item['choices']])

            if not is_test:
                program = item['program']
                if ADD_TRAIN_ANSWER:
                    program += item['answer']
                programs.append(program)
                answers.append(vocab['answer_token_to_idx'].get(item['answer']))

                item_functions = item['function']
                if IS_FUNC_TOKENIZER:
                    function = ' '.join(item_functions)
                else:
                    # 不分类时，使用序列生成，可以不给function编码和pad
                    function = [vocab['function_token_to_idx'][FuncTokensType.FUNC_START.value]] + \
                               [vocab['function_token_to_idx'][func] for func in item_functions] + \
                               [vocab['function_token_to_idx'][FuncTokensType.FUNC_END.value]]
                    pad_len = (max_func - len(function))
                    function = function + [vocab['function_token_to_idx'][FuncTokensType.FUNC_PAD.value]] * pad_len
                    function_mask = [1] * len(function) +\
                                    [vocab['function_token_to_idx'][FuncTokensType.FUNC_PAD.value]] * pad_len
                    if len(function) != len(function_mask):
                        logger.info("len function_mask", len(function_mask))
                        logger.info("len fuction", len(function))
                    function_masks.append(function_mask)

                functions.append(function)
                # program, pro_sequence, pro_functions, pro_inputs = self._get_program_seq(program)
                item_inputs = item['inputs']
                input_str = self.get_joint_inputs(item_inputs)
                inputs.append(input_str)

                # 构造cl负样本数据
                pro_sequence = [func + f" {ProgramToken.INPUT.value} {f' {ProgramToken.INPUT.value} '.join(inp)}" if inp else func
                                for func, inp in zip(item_functions, item_inputs)]
                if self.conf.replace_method == FunctionInputReplaceStrategy.FUNC.value:
                    ...
                elif self.conf.replace_method == FunctionInputReplaceStrategy.INPUT.value:
                    ...
                elif self.conf.replace_method == FunctionInputReplaceStrategy.OR.value:
                    target_random = [
                        ps if np.random.random() > self.conf.replace_value else np.random.choice(pro_sequence)
                        for ps in pro_sequence]
                    target_random_seq = f' {ProgramToken.FUNC.value} '.join(target_random)
                    target_random_seqs.append(target_random_seq)
                else:
                    raise ValueError(f'{self.conf.replace_method} Not defined!')

        sequences = questions + programs  # 163
        logger.info('Start tokenizing questions + programs')
        encoded_inputs = tokenizer(sequences, padding=True)
        logger.info('questions + programs tokenize ended.')
        logger.info(f'questions + programs tokenize keys{encoded_inputs.keys()}')

        logger.info(encoded_inputs['input_ids'][0])
        logger.info(tokenizer.decode(encoded_inputs['input_ids'][0]))
        logger.info(tokenizer.decode(encoded_inputs['input_ids'][-1]))
        max_seq_length = len(encoded_inputs['input_ids'][0])
        assert max_seq_length == len(encoded_inputs['input_ids'][-1])
        logger.info("questions + programs max length: {}".format(max_seq_length))
        return questions, programs, choices, answers, functions, function_masks, inputs, max_seq_length, target_random_seqs

    def _filter_new_data(self, dataset, is_random=False):
        if not self.is_filter_new_data:
            return dataset
        new_dataset, question2infos = [], {}
        for info in dataset:
            question2infos.setdefault(info['question'], []).append(info)
        for question, question_items in question2infos.items():
            if len(question_items) > self.filter_new_data_num:
                filter_data = random.sample(question_items, self.filter_new_data_num) if is_random else question_items[: self.filter_new_data_num]
                new_dataset.extend(filter_data)
            else:
                new_dataset.extend(question_items)
        return new_dataset

    @staticmethod
    def get_joint_inputs(inputs):
        input_sequences = []
        for i, inp in enumerate(inputs):
            if not inp:
                input_null = f'{ProgramToken.INPUT_NULL.value} ' if i == 0 else f' {ProgramToken.INPUT_NULL.value} '
                input_sequences.append(input_null)
            else:
                input_sequences.append(f' {ProgramToken.INPUT.value} '.join(inp))
        # f'{ProgramToken.FUNC.value}'.join(map(lambda x: f'{ProgramToken.INPUT.value}'.join(x), filter(None, inputs)))
        return f' {ProgramToken.FUNC.value} '.join(input_sequences)

    @staticmethod
    def _get_program_seq(program):
        pro_sequence, pro_functions, pro_inputs = [], [], []
        for item in program:
            func = item['function']
            inputs = item['inputs']
            args = ''
            for input in inputs:
                args += f' {ProgramToken.INPUT.value} ' + input
            pro_sequence.append(func + args)
            pro_functions.append(func)
            pro_inputs.append(inputs)
            # seq.append(func + '(' + '<c>'.join(inputs) + ')')
        tar_seq = f' {ProgramToken.FUNC.value} '.join(pro_sequence)
        if ADD_TRAIN_ANSWER:
            tar_seq += DELIMITER + ' '
        return tar_seq, pro_sequence, pro_functions, pro_inputs

    @staticmethod
    def _load_json_file(file_path, mode='r'):
        with open(file_path, mode=mode) as f:
            return json.load(f)

    def _save_vocab(self, vocab):
        for k in vocab:
            logger.info('{}:{}'.format(k, len(vocab[k])))

        work_dir_obj = Path(self.work_dir)
        if not work_dir_obj.exists():
            work_dir_obj.mkdir(parents=True)

        fn = work_dir_obj.joinpath('vocab.json')
        with open(fn, 'w') as f:
            json.dump(vocab, f, indent=2)
        logger.info('Dump vocab to {}'.format(fn))


if __name__ == '__main__':
    from conf.config import ROOT_DIR
    from conf.setting import BartSetting
    BartPreprocess(
        input_dir=ROOT_DIR.joinpath('data/dataset'),
        model_path=ROOT_DIR.joinpath('data/model/KQAPro_ckpt/program_ckpt'),
        work_dir=BartSetting.work_dir
    ).do()