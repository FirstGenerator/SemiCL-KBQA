from enum import Enum
from pathlib import Path

# # 路径
ROOT_DIR = Path(__file__).resolve().parent.parent

# # 日志
# LOG_LEVEL = 'INFO'
LOG_DIR = ROOT_DIR.joinpath('logs')
LOG_LEVEL = 'DEBUG'
LOG_FILE_HANDLER = 'RotatingFileHandler'
IS_DEBUG_MODE = True

DEVICE_NUM = 2
IS_NLTK_DATA_LOCAL = True
NLTK_DATA = str(ROOT_DIR.joinpath('data/nltk_data'))

ADD_TRAIN_ANSWER = False
DELIMITER = '<answer>'


class ProgramToken(Enum):
    FUNC = '<func>'
    INPUT = '<arg>'
    INPUT_SEP = '<input>'
    INPUT_NULL = '<unk>'


class FuncTokensType(Enum):
    FUNC_PAD = '<PAD>'
    FUNC_START = '<START>'
    FUNC_END = '<END>'


INIT_FUNC_TOKENS = {
    FuncTokensType.FUNC_PAD.value: 0,
    FuncTokensType.FUNC_START.value: 1,
    FuncTokensType.FUNC_END.value: 2,
}
IS_FUNC_TOKENIZER = True
IS_ADD_FUNC_TOKENIZER = False

STANFORD_NLP_MODEL = str(ROOT_DIR.joinpath('data/model/stanford-corenlp-full-2018-02-27'))


class ModelNameOption(Enum):
    pre_train = 'pre-train'
    fine_tune = 'fine-tune'
    func = 'func'


MODEL_NAME = 'no'  # pre-train, fine-tune, func


# 可视化配置
ENV = u'KQAPro'  # 虚拟环境
USE_INCOMING_SOCKET = False
OTHER_CONFIG = {
    'server': '127.0.0.1',
    'port': 80917,
}

IS_NEW_DATA = False
IS_FILTER_NEW_DATA = True
FILTER_NEW_DATA_NUM = 10
NEW_DATASET_NAME = ['train']

# WEBQSP配置
IS_WEBQSP_GRAIL = True
IS_WEBQSP = True

# 对逻辑表达式进行过滤
WEBQSP_BAD_ANSWERS = [None, 'null']
IS_FILTER_WEBQSP_ANSWERS = True

