import os
import json
import copy
import random
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import logger
from conf.setting import BartSetting
from manager.train import BartTrain
from manager.preprocess import BartPreprocess


class TrainKFold:

    def __init__(self, k_fold=5, config=BartSetting(), shuffle=True):
        self.k_fold = k_fold
        self.config = config
        self.data_sharding = KFold(n_splits=k_fold, shuffle=shuffle, random_state=self.config.seed)
        self.datasets = []
        self.json_file_name = {'train.json', 'val.json', 'test.json', 'kb.json'}
        self.save_data_dir = Path(self.config.work_dir).joinpath('datasets')
        self.work_base_dir = config.work_dir.joinpath('models')
        self.val_path = os.path.join(self.config.input_dir, 'val.json')
        if not self._check_date(self.save_data_dir):
            self._split_data()
        self.models = []

    def _split_data(self):
        logger.info(f'Data save path {str(self.save_data_dir)}.')
        logger.info(f'Start {self.k_fold}-fold cross-validation split data.')

        kb = json.load(open(os.path.join(self.config.input_dir, 'kb.json')))
        train_set = json.load(open(os.path.join(self.config.input_dir, 'train.json')))
        random.shuffle(train_set)
        train_set = np.array(train_set)
        val_set = json.load(open(self.val_path))
        count = 1
        for train_index, test_index in self.data_sharding.split(train_set):
            train_data, val_data = train_set[train_index].tolist(), train_set[test_index].tolist()
            logger.info(f'{count}-fold: train size {len(train_data)}, val size {len(val_data)}')
            data_dir = self.save_data_dir.joinpath(f'k_ford_{count}')
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
            self.datasets.append(str(data_dir))
            self._write_to_json(data_dir.joinpath('train.json'), train_data)
            self._write_to_json(data_dir.joinpath('val.json'), val_data)
            self._write_to_json(data_dir.joinpath('test.json'), val_set)
            self._write_to_json(data_dir.joinpath('kb.json'), kb)
            count += 1
        logger.info(f'End {self.k_fold}-fold cross-validation split data.')

    def fit(self):
        for k, input_dir in enumerate(self.datasets):
            config = copy.deepcopy(self.config)
            config.work_dir = self.work_base_dir.joinpath(f'{k}-ford')

            existed_bart_program_train_result = os.path.join(config.work_dir, 'Bart_train.txt')
            if os.path.isfile(existed_bart_program_train_result):
                os.remove(existed_bart_program_train_result)

            print("self.work_base_dir:", self.work_base_dir)
            print("config.work_dir:", config.work_dir)
            preprocess = BartPreprocess(input_dir, config.preprocess_model_path, config.work_dir)
            bp_train = BartTrain(config, preprocess)
            bp_train.do(sample=True)
            self.models.append(bp_train)
        return self.models

    def eval(self):
        if self.models:
            return
        for obj in self.work_base_dir.rglob('*checkpoint*'):
            config = copy.deepcopy(self.config)
            config.work_dir = obj.parent
            preprocess = BartPreprocess(input_dir=None, model_path=config.preprocess_model_path, work_dir=config.work_dir)
            bp_train = BartTrain(config, preprocess)
            self.models.append(bp_train)
        return

    def _predict(self):
        results = []
        for model in self.models:
            acc, result = model.eval(self.val_path)
            logger.info(f'acc: {acc}, model path: {model.conf.work_dir}')
            results.append(result)
        return results

    @staticmethod
    def _write_to_json(path, data):
        with open(path, "w") as fp:
            json.dump(data, fp, ensure_ascii=False)

    def _check_date(self, data_dir):
        """
        :param data_dir:
        :return:
        """
        if not data_dir.exists():
            return False
        dir_obj = Path(data_dir)
        childes = [obj for obj in dir_obj.iterdir()]
        if len(childes) != self.k_fold:
            return False
        for sub_obj in childes:
            sub_name_set = set([obj.name for obj in sub_obj.iterdir()])
            if not sub_name_set.issubset(self.json_file_name):
                return False
        self.datasets = childes
        return True


if __name__ == '__main__':
    split = TrainKFold()
    split.fit()
