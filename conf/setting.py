from enum import Enum
from conf.config import ROOT_DIR
from utils.base_setting import BaseSetting


class FunctionInputReplaceStrategy(Enum):
    FUNC = 'func'  # 只替换function
    INPUT = 'input'  # 只替换input
    OR = 'or'  # function和input为一组替换


class BartModelStrategy(Enum):
    CONTRASTIVE_LEARNING = 'cl'  # contrastive learning 进行对比学习
    KL_DIVERGENCE = 'kl'  # kl散度计算
    ROUTINE = 'routine'  # 常规训练
    CL_AND_KL = 'cl_kl'  # 同时进行对比学习和kl散度计算
    FUNC_CLASS = 'func_class'
    TREE_INFO = 'tree'
    PURE_CL = 'pure_cl'


class BartSetting(BaseSetting):
    train_result_method = 'best'  # all保留所有训练过程，best保留acc最高的
    model_strategy = BartModelStrategy.CONTRASTIVE_LEARNING.value  # 训练bart的策略

    replace_method = FunctionInputReplaceStrategy.OR.value  # 数据生成时，function和input打乱方式
    replace_value = 0.3

    model_describe = 'webqsp training'
    work_dir_suffix = '_webqsp_du_multi'
    input_dir = ROOT_DIR.joinpath('data/dataset_webqsp/')
    work_dir = ROOT_DIR.joinpath('data/output_dir/Bart_Program/work_dir{}'.format(work_dir_suffix))
    # save_dir = ROOT_DIR.joinpath('logs')  # 'path to save checkpoints and logs'
    kl_rate = 1  # 添加kl的权重
    tao = 0.18  # 对比学的的参数
    cl_rate = 0.8
    model_pair_rate = 1  # 进行对比学习，正负样本model训练的loss之和所占比例
    patience = 10

    max_func_size = 15
    func_input_size = 768

    is_again_train = True  # 训练中断后，使用中断时保存好的模型进行训练， 使用ckpt_dir保存训练好的模型
    is_load_again_train_info = False  # 是否恢复step信息
    preprocess_model_path = ROOT_DIR.joinpath('data')
    ckpt_dir = str(ROOT_DIR.joinpath(''))
    save_name = 'Bart_train-eval-KL-0.8'

    # training parameters
    weight_decay = 1e-5
    batch_size = 28
    seed = 666  # random seed
    learning_rate = 3e-5
    num_train_epochs = 100
    save_steps = 448
    logging_steps = 448
    # Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.
    warmup_proportion = 0.1
    adam_epsilon = 1e-8  # Epsilon for Adam optimizer.
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps = 1.0
    max_grad_norm = 1.0  # Max gradient norm.
    num_eval_epochs=100
    # validating parameters
    # num_return_sequences = 1
    # top_p =

    # model hyperparameters
    dim_hidden = 1024
    alpha = 1e-4

    warmup_steps = None

