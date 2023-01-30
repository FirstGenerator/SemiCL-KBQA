import os
import copy
import json
import math
import torch
import shutil
import traceback
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List
from types import SimpleNamespace
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score
from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration

from utils.logger import logger
from conf.setting import BartModelStrategy
from conf.config import (
    MODEL_NAME,
    ModelNameOption,
    FuncTokensType,
    INIT_FUNC_TOKENS,
    IS_FUNC_TOKENIZER,
    IS_WEBQSP_GRAIL,
    IS_WEBQSP,
)
from utils.visdom_client import visdom_client
from conf.data_config import DATA_MAP, BaseData

if MODEL_NAME == ModelNameOption.pre_train.value:
    from transformers import BartForConditionalGeneration as BartModel
else:
    from transformers import BartForConditionalGeneration as BartModel
import warnings

warnings.simplefilter("ignore")

OTHER_CONDITION = [
    BartModelStrategy.CONTRASTIVE_LEARNING.value,
    BartModelStrategy.PURE_CL.value
]


class BartTrain(Train):

    def __init__(self, conf, preprocess: Preprocess):
        self.conf = conf
        self.preprocess = preprocess
        self.device = get_device()
        self.train_loader = None
        self.val_loader = None
        self.kb = None
        self.rule_executor = None
        self.tokenizer = None
        self.model_path = None
        self.model = None
        self.vocab = None
        self.result = []
        self.data_type = DATA_MAP.get(self.conf.model_strategy, BaseData)

        self.other_model = None
        self.other_condition = OTHER_CONDITION
        self.is_other = self.conf.model_strategy in self.other_condition

        self.con_loss = ContrastiveLoss(self.conf.tao)
        self.best_acc = 0
        self.global_acc = 0
        self.patience_count = 0
        self.epoch2loss = {}

        self.embed_tokens = None
        self.embed_scale = None

        self.bart_config = None

        self.classifier = nn.Linear(self.conf.func_input_size, self.conf.max_func_size, bias=False)
        self.loss_func = nn.CrossEntropyLoss()

    def do(self):
        # 数据初始化
        logger.info('初始化')
        if not list(self.conf.work_dir.rglob('*.pt')):
            self.preprocess.do()
        # args display
        for k, v in vars(self.conf).items():
            logger.info(k + ':' + str(v))
        seed_everything(self.conf.seed)
        try:
            self.load_resource()
            self.train()
        except Exception as e:
            logger.error('An error occurred during training {}. error info:\n{}'.format(e, traceback.format_exc()))

    def load_resource(self):
        logger.info("Create train_loader and val_loader.........")
        vocab_json = os.path.join(self.conf.work_dir, 'vocab.json')
        train_pt = os.path.join(self.conf.work_dir, 'train.pt')
        val_pt = os.path.join(self.conf.work_dir, 'val.pt')
        self.train_loader = DataLoader(vocab_json, train_pt, self.conf.batch_size, self.conf.model_strategy, training=True)
        self.val_loader = DataLoader(vocab_json, val_pt, self.conf.batch_size, self.conf.model_strategy)
        # self.classifier = nn.Linear(hidden.size, num_class, bias=False)

        self.model_path = self.conf.ckpt_dir if self.conf.is_again_train else self.conf.work_dir

        self.vocab = self.train_loader.vocab
        logger.info("Create model.........")
        # BartConfig.decoder_layers = 2
        # BartConfig.encoder_layers = 2
        BartConfig.func_len = 30
        BartConfig.inp_len = 141
        BartConfig.seq_len = 153
        BartConfig.is_att_regex = True
        config_class, model_class, tokenizer_class = (BartConfig, BartModel, BartTokenizer)
        config = {
            'output_hidden_states': True,
            # 'is_encoder_decoder': True
        }
        self.tokenizer = tokenizer_class.from_pretrained(os.path.join(self.model_path, 'tokenizer'))
        model = model_class.from_pretrained(os.path.join(self.model_path, 'model'), **config)
        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        # model = model_class.from_pretrained(args.model_name_or_path)
        self.model = model.to(self.device)

        if MODEL_NAME == ModelNameOption.fine_tune.value:
            self.fixed_parameter(self.model)  # 固定参数

        if self.is_other:
            # other_model = model_class.from_pretrained(os.path.join(self.model_path, 'model'), **config)
            other_model = copy.deepcopy(self.model)
            self.other_model = other_model.to(self.device)
            # self.classifier = self.classifier.to(self.device)

        save_conf_path = self.conf.work_dir.joinpath('Setting.json')
        with open(save_conf_path, 'w') as f:
            json.dump(self.conf.to_dict(), f, cls=JsonEncoder)
        logger.info('Save train parameter path: {}'.format(save_conf_path))

    def load_bart_config(self):
        with open(self.conf.work_dir.joinpath('model/config.json')) as f:
            bart_config = SimpleNamespace(**json.load(f))
        return bart_config

    def train(self):
        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(self.train_loader) // self.conf.gradient_accumulation_steps * self.conf.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        bart_param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.conf.weight_decay, 'lr': self.conf.learning_rate},
            {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': self.conf.learning_rate}
        ]
        self.conf.warmup_steps = int(t_total * self.conf.warmup_proportion)
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.conf.learning_rate, eps=self.conf.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.conf.warmup_steps,
            num_training_steps=t_total
        )
        # Check if saved optimizer or scheduler states exist
        if os.path.isfile(os.path.join(self.model_path, "optimizer.pt")) and \
                os.path.isfile(os.path.join(self.model_path, "scheduler.pt")):
            # Load in optimizer and scheduler states
            map_location = self.device if not torch.cuda.is_available() else None
            optimizer.load_state_dict(torch.load(os.path.join(self.model_path, "optimizer.pt"), map_location=map_location))
            scheduler.load_state_dict(torch.load(os.path.join(self.model_path, "scheduler.pt")))

            # Train!
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(self.train_loader.dataset))
            logger.info("  Num Epochs = %d", self.conf.num_train_epochs)
            logger.info("  Gradient Accumulation steps = %d", self.conf.gradient_accumulation_steps)
            logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if self.conf.is_load_again_train_info and "checkpoint" in str(self.model_path):
            # set global_step to gobal_step of last saved checkpoint from model path
            global_step = int(self.model_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(self.train_loader) // self.conf.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(self.train_loader) // self.conf.gradient_accumulation_steps)
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        # if self.is_other:
        #     other_optimizer = copy.deepcopy(optimizer)
        #     other_scheduler = copy.deepcopy(scheduler)

        logger.info('Checking...')
        logger.info("===================Dev==================")
        logger.info('Train Info: {}'.format(self.conf.model_strategy))
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        pane_prefix = f"{self.conf.model_strategy}_{datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')}"
        loss_pane, acc_pane = f'{pane_prefix}_train_loss', f'{pane_prefix}_eval_acc'
        step_num = len(self.train_loader)
        for epoch in range(int(self.conf.num_train_epochs)):
            pbar = ProgressBar(n_total=len(self.train_loader), desc='Training')
            for step, batch in enumerate(self.train_loader):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                self.model.train()
                # batch = tuple(t.to(self.device) for t in batch)
                pad_token_id = self.tokenizer.pad_token_id
                source_ids, source_mask, y = batch[self.data_type.source_ids.value], batch[self.data_type.source_mask.value], batch[self.data_type.target_ids.value]
                y_ids, lm_labels = self.build_labels(y, pad_token_id)
                inputs = {
                    "input_ids": source_ids.to(self.device),  # torch.Size([32, 163])
                    "attention_mask": source_mask.to(self.device),
                    "decoder_input_ids": y_ids.to(self.device),  # torch.Size([32, 162])
                    "labels": lm_labels.to(self.device),  # torch.Size([32, 162])
                }

                if self.conf.model_strategy == BartModelStrategy.KL_DIVERGENCE.value:
                    outputs = self.model(**inputs)
                    kl_value = self.get_kl(outputs)
                    loss = outputs[0] + self.conf.kl_rate * kl_value

                elif self.conf.model_strategy == BartModelStrategy.CONTRASTIVE_LEARNING.value:
                    self.other_model.train()
                    inputs_pair_a = inputs
                    inputs_pair_a["output_attentions"] = True

                    y_cl = batch[self.data_type.tar_mask_ids.value]
                    y_cl_ids, lm_labels_cl = self.build_labels(y_cl, pad_token_id)
                    # y_cl_ids = y_cl[:, :-1].contiguous()
                    inputs_pair_b = {
                        "input_ids": source_ids.to(self.device),
                        "attention_mask": source_mask.to(self.device),
                        "decoder_input_ids": y_cl_ids.to(self.device),
                        "labels": lm_labels_cl.to(self.device),
                        "output_attentions": True,
                    }
                    outputs_pair_a = self.model(**inputs_pair_a)
                    outputs_pair_b = self.other_model(**inputs_pair_b)
                    # other_loss = outputs_pair_b[0]
                    model_loss = outputs_pair_a[0]
                    cl_loss = self.get_contrastive_learning_loss(outputs_pair_a, outputs_pair_b)
                    loss = self.conf.model_pair_rate * model_loss + self.conf.cl_rate * cl_loss

                elif self.conf.model_strategy == BartModelStrategy.PURE_CL.value:
                    inputs_pair_a = inputs

                    y_cl = batch[self.data_type.tar_mask_ids.value]
                    y_cl_ids = y_cl[:, :-1].contiguous()
                    inputs_pair_b = {
                        "input_ids": source_ids.to(self.device),
                        "attention_mask": source_mask.to(self.device),
                        "decoder_input_ids": y_cl_ids.to(self.device),
                        "labels": lm_labels.to(self.device),
                    }
                    outputs_pair_a = self.model(**inputs_pair_a)
                    outputs_pair_b = self.other_model(**inputs_pair_b)
                    loss = self.get_contrastive_learning_loss(outputs_pair_a, outputs_pair_b)

                elif self.conf.model_strategy == BartModelStrategy.CL_AND_KL.value:
                    inputs_pair_a = inputs
                    inputs_pair_b = {
                        "input_ids": source_ids.to(self.device),
                        "attention_mask": source_mask.to(self.device),
                        "labels": lm_labels.to(self.device),
                    }
                    outputs_pair_a = self.model(**inputs_pair_a)
                    outputs_pair_b = self.model(**inputs_pair_b)
                    model_loss = outputs_pair_a[0]
                    kl_value = self.get_kl(outputs_pair_a)
                    cl_loss = self.get_contrastive_learning_loss(outputs_pair_a, outputs_pair_b)

                    loss = self.conf.model_pair_rate * model_loss + self.conf.kl_rate * kl_value + self.conf.cl_rate * cl_loss

                elif self.conf.model_strategy == BartModelStrategy.ROUTINE.value:
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                elif self.conf.model_strategy == BartModelStrategy.FUNC_CLASS.value:
                    func_ids, func_masks = batch[self.data_type.function_ids.value], batch[self.data_type.function_masks.value]
                    input_ids, input_masks = batch[self.data_type.inputs_ids.value], batch[self.data_type.input_masks.value]
                    func_pad_id = pad_token_id if IS_FUNC_TOKENIZER else INIT_FUNC_TOKENS[FuncTokensType.FUNC_PAD.value]
                    func_ids, lm_func_labels = self.build_labels(func_ids, func_pad_id)
                    input_ids, lm_input_labels = self.build_labels(input_ids, pad_token_id)
                    func_source_masks = self.build_masks(func_masks)
                    input_source_masks = self.build_masks(input_masks)
                    inputs.update({
                        # "input_ids": source_ids.to(self.device),
                        # "attention_mask": source_mask.to(self.device),
                        # "labels": lm_labels.to(self.device),
                        'decoder_func_ids': func_ids.to(self.device),
                        'func_labels': lm_func_labels.to(self.device),
                        'decoder_inp_ids': input_ids.to(self.device),
                        'input_labels': lm_input_labels.to(self.device),
                        "decoder_attention_func_mask": func_source_masks.to(self.device),
                        "decoder_attention_input_mask": input_source_masks.to(self.device)
                    })
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                elif self.conf.model_strategy == BartModelStrategy.TREE_INFO.value:
                    tree_data = batch[self.data_type.tree_data.value].to(self.device)

                    input_ids = inputs.pop('input_ids')
                    inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale + tree_data

                    inputs['inputs_embeds'] = inputs_embeds
                    outputs = self.model(**inputs)
                    loss = outputs[0]

                else:
                    raise ValueError(f'{self.conf.model_strategy} Not defined!')
                self.epoch2loss.setdefault(epoch, []).append(loss.item())
                loss.backward()
                self.set_line_chart(
                    x=torch.Tensor([epoch*step_num + step]),
                    y=loss.unsqueeze(dim=0),
                    win=loss_pane,
                    update='append',
                    title=loss_pane
                )
                pbar(step, {'loss': loss.item()})
                tr_loss += loss.item()
                if (step + 1) % self.conf.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.conf.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    # if self.is_other:
                    #     # if not self.conf.model_strategy == BartModelStrategy.FUNC_CLASS.value:
                    #     #     other_loss.backward()
                    #     torch.nn.utils.clip_grad_norm_(self.other_model.parameters(), self.conf.max_grad_norm)
                    #     other_optimizer.step()
                    #     other_scheduler.step()  # Update learning rate schedule
                    #     self.other_model.zero_grad()

            acc = self.validate(epoch, acc_pane)
            self.proceed_save_model(self.conf.train_result_method, acc, global_step, optimizer, scheduler)
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

            if acc >= self.global_acc:
                self.patience_count = 0
                self.global_acc = acc
            else:
                self.patience_count += 1
            if self.patience_count >= self.conf.patience:
                return global_step, tr_loss / global_step
        return global_step, tr_loss / global_step

    def validate(self, epoch, acc_pane=None):
        self.model.eval()
        count, correct, true_fun_count, true_inp_count, true_answer_count = 0, 0, 0, 0, 0
        with torch.no_grad():
            all_outputs, all_answers, all_true_program = [], [], []
            all_functions, all_true_functions = [], []
            for batch in tqdm(self.val_loader, total=len(self.val_loader)):
                source_ids, target_ids = batch[self.data_type.source_ids.value].to(self.device), batch[self.data_type.target_ids.value].to(self.device)
                if not IS_WEBQSP_GRAIL:
                    answer = batch[self.data_type.answers.value].to(self.device)
                # source_ids, source_mask, choices, target_ids, functions, answer = [x.to(self.device) for x in batch]
                outputs = self.model.generate(
                    input_ids=source_ids,
                    max_length=500,
                )
                all_true_program.extend(target_ids.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
                if not IS_WEBQSP_GRAIL:
                    all_answers.extend(answer.cpu().numpy())

            outputs = [self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output_id in all_outputs]
            true_outputs = [self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output_id in all_true_program]
            if not IS_WEBQSP_GRAIL:
                given_answer = [self.val_loader.vocab['answer_idx_to_token'][a] for a in all_answers]
            else:
                given_answer = outputs # webqsp数据暂不需要，使用outputs替代，实际未调用
            rule_answers,  program_infos, result_list = [], [], []
            idx = 0
            for a, output, true_output in tqdm(zip(given_answer, outputs, true_outputs)):
                if IS_WEBQSP_GRAIL:
                    # logger.info('output:{}'.format(output))
                    # logger.info('true_output:{}'.format(true_output))

                    if output == true_output:
                        correct += 1
                    count += 1

                    result_dict = force_top_1_eval(idx, output, true_output)
                    result_list.append(result_dict)

                    idx += 1

                else:
                    func_list, inputs_list, answer = get_function_and_input(output)
                    true_func_list, true_inputs_list, true_answer = get_function_and_input(true_output)

                    match = False
                    if func_list == true_func_list:
                        true_fun_count += 1
                        match = True

                    if match and inputs_list == true_inputs_list:
                        true_inp_count += 1
                    if answer == true_answer:
                        true_answer_count += 1

                    ans, lst = self.rule_executor.forward(func_list, inputs_list, ignore_error=True)
                    program_infos.append({
                        'func': true_func_list,
                        'pred_func': func_list,
                        'match_func': match,
                        'input': true_inputs_list,
                        'pred_input': inputs_list,
                        'match_input': inputs_list == true_inputs_list,
                        'answer': a,
                        'pred_answer': ans,
                        'match_answer': a == ans,
                    })
                    if ans != a:
                        # logger.info(colored(output, 'red'))
                        # logger.info(func_list)
                        # logger.info(inputs_list)
                        ...
                    if ans == None:
                        ans = 'no'
                    if ans == a:
                        correct += 1
                    count += 1
                    rule_answers.append(ans)

            acc = correct / count

            if IS_WEBQSP:
                eval_webqsp(result_list)
                result = {
                    'epoch': epoch,
                    'count': count,
                    'correct': correct,
                    'Accuracy': acc,
                }
                logger.info(f"\n\n>>> result:\n{result}")
            else:
                result = {
                    'epoch': epoch,
                    'count': count,
                    'match_fun_num': true_fun_count,
                    'match_inp_num': true_inp_count,
                    'true_answer_num': true_answer_count,
                    'function_ratio': true_fun_count / count,
                    'inputs_ratio': true_inp_count / count,
                    'answer_ratio': true_answer_count / count,
                    'correct': correct,
                    'Accuracy': acc,
                }
                logger.info(f"\n\n>>> result:\n{result}")
            # if acc_pane:
            #     self.set_line_chart(
            #         x=torch.Tensor([epoch]),
            #         y=torch.Tensor([[acc, result['function_ratio'], result['inputs_ratio']]]),
            #         win=acc_pane,
            #         update='append',
            #         title=acc_pane, xlabel='epoch', legend=['acc', 'func', 'input']
            #     )
            # logger.info('acc: {}'.format(acc))
            # result['true_answer'] = given_answer
            # result['rule_answers'] = rule_answers
            # result['loss'] = self.epoch2loss[epoch]
            # result['program_infos'] = program_infos
            # if all([all_true_functions, all_true_functions]):
            #     func_acc = accuracy_score(all_true_functions, all_functions)
            #     result['func_acc'] = func_acc
            #     logger.info('Func eval acc: {}'.format(func_acc))
            # # self.result.append(result)
            # save_result(result, 'Bart_train', self.conf.work_dir, mode='a+', file_type='txt')
            return acc

    def eval(self):
        self.load_resource()
        return self.validate(-1)

    def proceed_save_model(self, method, acc, global_step, optimizer, scheduler):
        output_dir = os.path.join(self.conf.work_dir, "acc_{}_checkpoint-{}".format(acc, global_step))
        if method == 'all':
            self.save_model(output_dir, optimizer, scheduler)

        elif method == 'best':
            if acc < self.best_acc:
                return
            old_outputs = Path(output_dir).parent.glob('acc*')
            for old_out_obj in old_outputs:
                shutil.rmtree(old_out_obj)
            self.best_acc = acc
            self.save_model(output_dir, optimizer, scheduler)
        else:
            raise ValueError(f'{method} not defined!')

    def save_model(self, output_dir, optimizer, scheduler):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        self.conf.ckpt_dir = output_dir
        model_to_save.save_pretrained(os.path.join(output_dir, 'model'))
        self.tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
        torch.save(self.conf, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)
        # tokenizer.save_vocabulary(output_dir)
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)
        logger.info("\n")

    # @staticmethod
    def get_kl(self, model_output):
        decoder_last_hidden_state = model_output.decoder_hidden_states[-1]
        encoder_last_hidden_state = model_output.encoder_last_hidden_state

        bath_size = encoder_last_hidden_state.size(0)
        encoder_len = encoder_last_hidden_state.size(1)
        decoder_len = decoder_last_hidden_state.size(1)
        transfer_tensor = torch.rand((bath_size, decoder_len, encoder_len))
        device = encoder_last_hidden_state.device
        transfer_tensor = transfer_tensor.to(device)
        transfered_encoder_feature = torch.matmul(transfer_tensor, encoder_last_hidden_state)

        kl_value = F.kl_div(
            transfered_encoder_feature.log_softmax(dim=-1),
            decoder_last_hidden_state.softmax(dim=-1)
        )
        return kl_value

    def get_contrastive_learning_loss(self, model_output_by_sample_a, model_output_by_sample_b):
        decoder_last_hidden_state_sample_a = model_output_by_sample_a.decoder_hidden_states[-1]
        decoder_last_hidden_state_sample_b = model_output_by_sample_b.decoder_hidden_states[-1]

        a = decoder_last_hidden_state_sample_a.sum(1)/decoder_last_hidden_state_sample_a.size(1)
        b = decoder_last_hidden_state_sample_b.sum(1)/decoder_last_hidden_state_sample_b.size(1)

        cl_loss = compute_loss(torch.cat((a, b)), self.device)

        # cl_loss = 0
        # for i in range(len(decoder_last_hidden_state_pair_p)):
        #     fea, pos_fea = decoder_last_hidden_state_pair_p[i], decoder_last_hidden_state_pair_n[i]
        #     neg_fea = torch.cat(
        #         (decoder_last_hidden_state_pair_p[:i], decoder_last_hidden_state_pair_p[i + 1:]),
        #         0
        #     )
        #     cl_loss += self.con_loss(fea, pos_fea, neg_fea)

        return cl_loss

    @staticmethod
    def fixed_parameter(model):
        for name, param in model.named_parameters(recurse=True):
            model_name = name.split('.')[0]
            if 'lm_head' != model_name and 'new_lm_head' != model_name:
                param.requires_grad = False

    @staticmethod
    def build_labels(labels, pad_token_id):
        y_ids = labels[:, :-1].contiguous()
        lm_labels = labels[:, 1:].clone()
        lm_labels[labels[:, 1:] == pad_token_id] = -100
        return y_ids, lm_labels

    @staticmethod
    def build_masks(origin_masks):
        source_masks = origin_masks[:, :-1].contiguous()
        target_masks = origin_masks[:, 1:].clone()
        return source_masks

    @staticmethod
    def set_line_chart(x, y, win, title, update='append', xlabel=None, ylabel=None, legend: List = None):
        if not visdom_client:
            return
        opts = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel}
        if legend:
            opts['legend'] = legend
        visdom_client.line(X=x, Y=y, win=win, update=update, opts=opts)


if __name__ == '__main__':
    from manager.preprocess import BartPreprocess
    from conf.setting import BartSetting

    conf = BartSetting()
    preprocess = BartPreprocess(conf.input_dir, conf.preprocess_model_path, conf.work_dir)

    bp_train = BartTrain(conf, preprocess)
    bp_train.do()
    # res = bp_train.eval()
    # file_name = conf.save_name
    # save_result(bp_train.result, file_name, bp_train.conf.work_dir, file_type='json')
