# -*- coding: utf-8 -*-
import os
import copy
import json
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import BartForConditionalGeneration, BartTokenizer
from manager.data import Dataset, collate

from utils.logger import logger
from utils.misc import get_device, verify_answers
from manager.Code_WebQSP.eval_prediction import get_entity_mapping_from_top_candidates, execute_normed_s_expr


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, batch_size, model_strategy=None, training=False):
        dataset = Dataset(question_pt, model_strategy)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate,
        )


class WebQsaDataAugmentationPreprocess:

    def __init__(
            self,
            model_path,
            train_pt_path,
            train_json_path,
            is_add_program_random=False,
            num_return_sequences=4,
            batch_size=32,
            epoch_num=200,
            sample_num=6,
            random_num=5,
    ):
        self.model_path = model_path
        self.train_pt_path = train_pt_path
        self.train_json_path = train_json_path

        self.is_add_program_random = is_add_program_random
        self.num_return_sequences = num_return_sequences
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.sample_num = sample_num
        self.random_num = random_num

        self.question2infos = self._get_train_question2infos()

        self.device = get_device()

    def preprocess(self):
        train_data = self._load_train_pt()
        model, tokenizer = self._load_model()
        model.train()  # 让dropout生效，获取不同的数据
        question2infos = {}
        with torch.no_grad():  # 不进行梯度计算
            for epoch in range(self.epoch_num):
                for batch in tqdm(train_data, total=len(train_data)):
                    input_ids, target_ids = batch['input_ids'], batch['decoder_input_ids']
                    outputs = model.generate(
                        input_ids=input_ids.to(self.device),
                        max_length=500,
                        temperature=1.2,
                        num_return_sequences=self.num_return_sequences,
                        do_sample=True
                    )
                    last_dim = outputs.size(-1)
                    outputs = outputs.reshape(-1, self.num_return_sequences, last_dim)
                    pred_outputs = [
                        [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            for output_id in output_ids] for output_ids in outputs
                    ]
                    questions = [
                        tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for input_id in input_ids
                    ]
                    true_outputs = [
                        tokenizer.decode(target_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for target_id in target_ids
                    ]
                    batch_question2infos = self._build_data(questions, true_outputs, pred_outputs)
                    self._update_data(question2infos, batch_question2infos)

        data = list(question2infos.values())
        self._get_statistics_info(data)
        self._save_data(data)

        data_infos = self._encode_dataset(data, tokenizer)
        self._save_train_pt(data_infos)
        return data_infos

    def build_external_data(self, data_items: List[Dict, ]):
        _, tokenizer = self._load_model()
        self._get_statistics_info(data_items)

        data_infos = self._encode_dataset(data_items, tokenizer)
        self._save_train_pt(data_infos)

    def build_data_enhancement_routine(self, data_items: List[Dict, ]):
        _, tokenizer = self._load_model()
        enhancement_data = []
        for data_item in data_items:
            for k, v in list(data_item.items()):
                if isinstance(v, list):
                    data_item[k] = list(set(v))
                # data_item['num'] = len(data_item['SExpr_positive'])
            for sql in data_item['SExpr_positive'][: self.sample_num + 1]:
                item = copy.deepcopy(data_item)
                item['SExpr'] = sql
                item['program'] = data_item['SExpr']
                enhancement_data.append(item)
            data_item['program'] = data_item['SExpr']
            enhancement_data.append(data_item)
        for _ in range(self.random_num):
            random.shuffle(enhancement_data)
        data_infos = self._encode_dataset(enhancement_data, tokenizer)
        self._save_train_pt(data_infos)

    @staticmethod
    def _encode_dataset(dataset, tokenizer):
        logger.info('Start encoding data......')

        questions = [item['question'] for item in dataset]
        input_ids = tokenizer.batch_encode_plus(questions, padding=True, truncation=True)
        source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
        source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)

        programs = [item['SExpr'] for item in dataset]
        target_ids = tokenizer.batch_encode_plus(programs, padding=True, truncation=True)
        target_ids = np.array(target_ids['input_ids'], dtype=np.int32)

        # null
        answers = np.array([0] * target_ids.shape[0], dtype=np.int32)
        choices = np.array([0] * target_ids.shape[0], dtype=np.int32)

        program_cls = [item['program'] for item in dataset]
        tar_mask_ids = tokenizer.batch_encode_plus(program_cls, padding=True, truncation=True)
        tar_mask_ids = np.array(tar_mask_ids['input_ids'], dtype=np.int32)

        data_infos = {
            # sequence data
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,

            # null
            'answers': answers,
            'choices': choices,

            # cl
            'tar_mask_ids': tar_mask_ids,
        }
        logger.info('Data encoding completed.')
        return data_infos

    @staticmethod
    def _update_data(question2infos: Dict, batch_question2infos: Dict):
        for question, info in batch_question2infos.items():
            global_info = question2infos.setdefault(question, {})
            global_info.setdefault('SExpr_positive', []).extend(info.get('SExpr_positive', []))
            global_info.setdefault('SExpr_negative', []).extend(info.get('SExpr_negative', []))
            global_info.setdefault('SExpr_random', []).extend(info.get('SExpr_random', []))
            global_info.setdefault('SExpr', info['SExpr'])
            global_info.setdefault('question', info['question'])

    def _get_statistics_info(self, data_items: List[Dict, ]):
        positive_num, negative_num, program_ran_num = 0, 0, 0
        for item in data_items:
            program_positive = item.get('SExpr_positive', [])
            negative_program = item.get('SExpr_negative', [])
            program_random = item.get('SExpr_random', [])

            if program_positive:
                program = random.choice(list(set(program_positive)))
            elif self.is_add_program_random:
                program = random.choice(list(set(program_random)))
            else:
                program = item['SExpr']
            item['program'] = program

            positive_num += len(set(program_positive))
            negative_num += len(set(negative_program))
            program_ran_num += len(set(program_random))
        logger.info(f"positive_num: {positive_num}")
        logger.info(f"negative_num: {negative_num}")
        logger.info(f"program_random: {program_ran_num}")
        return (positive_num, negative_num, program_ran_num), data_items

    @staticmethod
    def get_statistics_info_v2(data_items: List[Dict, ]):
        # 前面数据构建已处理过，增强的数据中的逻辑表达式与正确的相同的情况
        positive_num, negative_num = 0, 0
        question_positive_num, question_negative_num = 0, 0
        for item in data_items:
            program_positive = item.get('SExpr_positive', [])
            negative_program = item.get('SExpr_negative', [])

            question_positive_num += 1 if program_positive else 0
            question_negative_num += 1 if negative_program else 0

            positive_num += len(set(program_positive))
            negative_num += len(set(negative_program))
        logger.info(f">>> positive_num: {positive_num}")
        logger.info(f">>> negative_num: {negative_num}")
        logger.info(f">>> question_positive_num: {question_positive_num}")
        logger.info(f">>> question_negative_num: {question_negative_num}")

        return positive_num, negative_num

    def _build_data(self, questions: List[str, ], true_sql_list: List[str, ], pred_sqls_list: List[List[str, ], ]):
        question2infos = {}
        for question, true_sql, pred_sqls in zip(questions, true_sql_list, pred_sqls_list):
            is_match = False
            for pred_sql in set(pred_sqls):
                if true_sql != pred_sql:
                    result_dict = self._get_answers(question, pred_sql, true_sql)
                    if verify_answers(result_dict['Answers'], result_dict['true_Answers']):
                        question2infos.setdefault(question, {}).setdefault('SExpr_positive', []).append(pred_sql)
                        is_match = True
                    else:
                        question2infos.setdefault(question, {}).setdefault('SExpr_negative', []).append(pred_sql)

            if not is_match:
                question2infos.setdefault(question, {}).setdefault('SExpr_random', []).append(
                    self._get_cl_data_by_sexpr(true_sql)
                )
            question2infos[question].setdefault('SExpr', true_sql)
            question2infos[question].setdefault('question', question)

        return question2infos

    def _get_answers(self, question, output, true_output):
        val_data = self.question2infos[question]
        assert val_data['SExpr'].replace(' ,', ',') == true_output
        SExpr_ori = val_data['SExpr_ori']

        entity_label_map = get_entity_mapping_from_top_candidates(SExpr_ori)  # add

        lf, answers = execute_normed_s_expr(output, entity_label_map)
        true_lf, true_answers = execute_normed_s_expr(true_output, entity_label_map)
        result_dict = {
            'QuestionId': val_data['QuestionId'],
            'logical_form': lf,
            'Answers': answers,
            'true_logical_form': true_lf,
            'true_Answers': true_answers,
        }
        return result_dict

    def _get_cl_data_by_sexpr(self, sexpr: str):
        if not sexpr.startswith('('):  # 为null
            return sexpr

        left_bracket_min_num = 2
        if sexpr.count('(') <= left_bracket_min_num:  # 只有一个逻辑表达式
            return sexpr

        chars = sexpr.split()[1:-1]  # 去掉第一个()
        bracket_offset, index2sub_index = self._get_bracket_indexes(chars)
        index2random_index = self._get_index2random_index(list(index2sub_index.keys()))
        for index, random_index in index2random_index.items():
            chars[index[0]: index[1]] = chars[random_index[0]: random_index[1]]
        chars = [sexpr[0]] + chars + [sexpr[-1]]
        return ' '.join(chars)

    @staticmethod
    def _get_bracket_indexes(chars):
        bracket_start_indexes, bracket_offset, index2sub_index = [], [], {}

        sub_indexes = []
        for i, char in enumerate(chars):
            if char == '(':
                bracket_start_indexes.append(i)
            elif char == ')':
                start_index = bracket_start_indexes.pop()
                indexes = [start_index, i+1]
                bracket_offset.append(indexes)
                if len(bracket_start_indexes) != 0:
                    sub_indexes.append(indexes)
                else:
                    index2sub_index.setdefault((start_index, i+1), []).extend(sub_indexes)
                    sub_indexes = []

        assert not bracket_start_indexes, '栈必须为空'

        return bracket_offset, index2sub_index

    @staticmethod
    def _get_index2random_index(indexes):
        index2random_index = {}

        min_index_num = 2
        if len(indexes) < min_index_num:
            index2random_index[indexes[0]] = indexes[0]
            return index2random_index

        random_indexes = copy.deepcopy(indexes)
        while True:
            random.shuffle(random_indexes)
            if random_indexes != indexes:
                break

        for i, index in enumerate(indexes):
            index2random_index[index] = random_indexes[i]

        return index2random_index

    def _load_train_pt(self):
        loader = DataLoader(self.train_pt_path, self.batch_size, training=True)
        logger.info('Loading data is complete, and a total of {} pieces of data are loaded.'.format(len(loader)))
        return loader

    def _load_model(self):
        logger.info('Loading model......')
        model = BartForConditionalGeneration.from_pretrained(os.path.join(self.model_path, 'model'))
        tokenizer = BartTokenizer.from_pretrained(os.path.join(self.model_path, 'tokenizer'))
        logger.info('Model loading completed.')
        return model.to(self.device), tokenizer

    def _get_train_question2infos(self, mode='r', encoding='utf-8'):
        with open(self.train_json_path, mode=mode, encoding=encoding) as f:
            question2infos = {item['question']: item for item in json.load(f)}
        return question2infos

    def _save_train_pt(self, data, file_path=None):
        if not file_path:
            file_path = self.train_pt_path
        with open(file_path, 'wb') as f:
            for data_name, data_value in data.items():
                logger.info(f'{data_name} >>> {data_value.shape}')
            pickle.dump(data, f)
        logger.info('Succeeded in rewriting {}.'.format(self.train_pt_path))

    def _save_data(self, data):
        out_dir = Path(self.train_json_path).parent
        with open(out_dir.joinpath('data.json'), 'w') as f:
            json.dump(data, f)
        logger.info('Enhanced data saved successfully with file path {}'.format(out_dir))


if __name__ == '__main__':
    from conf.config import ROOT_DIR

    pro = WebQsaDataAugmentationPreprocess(
        model_path='',
        train_pt_path=ROOT_DIR.joinpath(''),
        train_json_path=ROOT_DIR.joinpath('data/dataset_webqsp/train.json'),
        epoch_num=200
    )
    pro.preprocess()
    # pro.build_external_data(data_items=data)
    # pro.build_data_enhancement_routine(data_items=data)

    # 1. question -> lf -> multi lf (positive, negative)
    ## 2. aug routine
    # 3. cl

