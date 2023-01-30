import os
import re
import torch
from tqdm import tqdm
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer

from manager.data import DataLoader
# from manager.executor_rule import RuleExecutor
from utils.logger import logger
from utils.misc import seed_everything, get_device, save_result, get_function_and_input

import warnings

warnings.simplefilter("ignore")  # hide warnings that caused by invalid sparql query


class BartPredict:

    def __init__(self, conf):
        self.conf = conf
        self.device = get_device()
        self.model = None
        self.tokenizer = None
        self.data = None
        self.executor = None
        seed_everything(self.conf.seed)
        self.load_resource()

    def do(self):
        for k, v in vars(self.conf).items():
            logger.info(k + ':' + str(v))
        return self.predict()

    def load_resource(self):
        logger.info("Create train_loader and val_loader.........")
        vocab_json = os.path.join(self.conf.work_dir, 'vocab.json')
        train_pt = os.path.join(self.conf.work_dir, 'train.pt')
        val_pt = os.path.join(self.conf.work_dir, 'test.pt')
        train_loader = DataLoader(vocab_json, train_pt, self.conf.batch_size, training=True)
        val_loader = DataLoader(vocab_json, val_pt, self.conf.batch_size)

        vocab = train_loader.vocab
        # kb = DataForSPARQL(os.path.join(self.conf.work_dir, 'kb.json'))
        logger.info("Create model.........")
        config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
        self.tokenizer = tokenizer_class.from_pretrained(os.path.join(self.conf.ckpt_dir, 'tokenizer'))
        model = model_class.from_pretrained(os.path.join(self.conf.ckpt_dir, 'model'))
        self.model = model.to(self.device)
        logger.info(model)
        rule_executor = RuleExecutor(vocab, os.path.join(self.conf.work_dir, 'kb.json'))
        self.data = val_loader
        self.executor = rule_executor
        # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
        # self.predict(val_loader, rule_executor)
        # vis(args, kb, model, val_loader, device, tokenizer)

    def post_process(self, text):
        pattern = re.compile(r'".*?"')
        nes = []
        for item in pattern.finditer(text):
            nes.append((item.group(), item.span()))
        pos = [0]
        for name, span in nes:
            pos += [span[0], span[1]]
        pos.append(len(text))
        assert len(pos) % 2 == 0
        assert len(pos) / 2 == len(nes) + 1
        chunks = [text[pos[i]: pos[i + 1]] for i in range(0, len(pos), 2)]
        for i in range(len(chunks)):
            chunks[i] = chunks[i].replace('?', ' ?').replace('.', ' .')
        bingo = ''
        for i in range(len(chunks) - 1):
            bingo += chunks[i] + nes[i][0]
        bingo += chunks[-1]
        return bingo

    def vis(self):
        while True:
            # text = 'Who is the father of Tony?'
            # text = 'Donald Trump married Tony, where is the place?'
            text = input('Input your question:')
            with torch.no_grad():
                input_ids = self.tokenizer.batch_encode_plus([text], max_length=512, pad_to_max_length=True,
                                                             return_tensors="pt",
                                                             truncation=True)
                source_ids = input_ids['input_ids'].to(self.device)
                outputs = self.model.generate(
                    input_ids=source_ids,
                    max_length=500,
                )
                outputs = [self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                           for
                           output_id in outputs]
                outputs = [self.post_process(output) for output in outputs]
                logger.info(outputs[0])

    def predict(self):
        self.model.eval()
        count, correct = 0, 0
        pattern = re.compile(r'(.*?)\((.*?)\)')
        with torch.no_grad():
            all_outputs = []
            for batch in tqdm(self.data, total=len(self.data)):
                batch = batch[:3]
                source_ids, source_mask, choices = [x.to(self.device) for x in batch]
                outputs = self.model.generate(
                    input_ids=source_ids,
                    max_length=500,
                )
                all_outputs.extend(outputs.cpu().numpy())
            result = []
            outputs = [self.tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output_id in all_outputs]
            out_file = os.path.join(self.conf.work_dir, 'predict.txt')
            with open(out_file, 'w') as f:
                for output in tqdm(outputs):
                    func_list, inputs_list, answer = get_function_and_input(output)
                    ans, lst = self.executor.forward(func_list, inputs_list, ignore_error=True)
                    result.append({
                        'predict_answer': ans,
                        'decode_answerr': answer,
                        'pred_function': func_list,
                        'pred_input': inputs_list,
                        'end_step': lst
                    })
                    if ans == None:
                        ans = 'no'
                    f.write(ans + '\n')
        save_result(result, 'BartProgPredict', self.conf.work_dir, file_type='json')
        return out_file


if __name__ == '__main__':
    from conf.setting import BartSetting
    pred = BartPredict(BartSetting)
    pred.do()
