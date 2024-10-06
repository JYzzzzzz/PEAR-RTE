import glob
import logging
import os
import sys
import json
import time
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from lib.callback.optimizater.adamw import AdamW
from lib.callback.lr_scheduler import get_linear_schedule_with_warmup
from lib.callback.progressbar import ProgressBar
from lib.tools.common import seed_everything, list_write_txt
from lib.tools.common import init_logger, logger

from lib.models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from lib.models.bert_for_ner import BertCrfForNer, BertLstmForNer
from lib.models.albert_for_ner import AlbertCrfForNer
from lib.processors.utils_ner import CNerTokenizer, get_entities
from lib.processors.ner_seq import convert_examples_to_features
from lib.processors.ner_seq import ner_processors as processors
from lib.processors.ner_seq import collate_fn, collate_fn_2310ForTask1
from lib.processors.ner_seq import t1_sent_part_li, t1_ent_li, des_encrypt, des_decrypt
from lib.metrics.ner_metrics import SeqEntityScore

from dataset_loader import Dataset, NER_TAG_LIST, ADDITIONAL_SPECIAL_TOKENS, \
    sent_token_cut, sent_add_prompt, ner_tag_decode, process_batch

from transformers import BertTokenizerFast

##### 数据集读取、格式转换、特殊符号定义（自定义库）

# from tools.finetuning_argparse import get_argparse

# MODEL_CLASSES = {
#     ## bert ernie bert_wwm bert_wwwm_ext
#     'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
#     'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
# }
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertLstmForNer, BertTokenizerFast),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}

HINT_RELA_SENT = '##ᄀ'  # '【关系提示句】'
HINT_ENTITIES_SENT = '##ᄁ'  # '【实体提示句】'
HINT_HEAD_ENTITY = '##ᄂ'  # '【头实体】'
HINT_TAIL_ENTITY = '##ᄃ'  # '【尾实体】'
HINT_ENTITY_END = '##ᄅ'  # '【实体结束】'
Mobile_Match_Special_Tokens = {
    'unk_tokens': ['￡', '…', '\uf06c', 'Φ', '屮', '\uf09e', '\xa0'],
    'token_trans_1to1': ((' ', '\n', '\t', '—', '–', '‐', '﹤', 'Ｒ', '‘', '’',
                          '“', '”', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                          'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'),
                         ('space', 'enter', 'tab', '-', '-', '-', '<', 'r', '##ᄆ', '##ᄇ',
                          '##ᄆ', '##ᄇ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                          'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                          's', 't', 'u', 'v', 'w', 'x', 'y', 'z')),
    'token_trans_nto1': ((['【', '壹', '】'], HINT_RELA_SENT), (['【', '贰', '】'], HINT_ENTITIES_SENT),
                         (['【', '埱', '】'], HINT_HEAD_ENTITY), (['【', '葳', '】'], HINT_TAIL_ENTITY),
                         (['【', '桀', '】'], HINT_ENTITY_END))
}


class Histogram:
    """
    直方图相关类

    1、初始化
    2、使用 input_one_data 一个一个添加数据
    3、
    """
    def __init__(self, left_lim, right_lim, interval, init_show: str = ""):
        """

        :param left_lim: 统计的左边界
        :param right_lim: 统计的右边界
        :param interval: 各区间的间隔。边界规则：[)，最后一个区间规则：[]
        :param init_show: 没啥用
        """
        self.statistic_info = []
        self.statistic_info_simple = []  # 直接显示这个即可
        left = left_lim  # 每一柱的左边界
        while left < right_lim:
            right = right_lim if left + interval >= right_lim else left + interval
            col_info = [left, right, 0, 0.]  # 左边界，右边界，个数，占比。!!!!!!!!!!!!!
            # 边界规则：[)，最后一个区间规则：[]
            col_info_simple = [round(left, 2), 0.]  # 左边界，占比
            self.statistic_info.append(col_info.copy())
            self.statistic_info_simple.append(col_info_simple.copy())
            left = right
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.sample_in_lim_num = 0
        self.larger_num = 0
        self.smaller_num = 0
        # print("-- a histogram has been initialized: {}".format(init_show))
        # print(self.statistic_info_simple)

    def input_one_data(self, data):  # 直方图统计时添加一个数据
        if data < self.left_lim:
            self.smaller_num += 1
            return
        elif data > self.right_lim:
            self.larger_num += 1
            return

        for i in range(len(self.statistic_info) - 1, -1, -1):
            if self.statistic_info[i][0] <= data <= self.statistic_info[i][1]:  # [l, r)
                self.statistic_info[i][2] += 1
                break

    def update_ratio(self):  # 直方图显示前更新比率
        sample_num = 0
        for col_info in self.statistic_info:
            sample_num += col_info[2]
        self.sample_in_lim_num = sample_num

        if sample_num <= 0:  # 防止零除错误
            sample_num = 1

        for i in range(len(self.statistic_info)):
            self.statistic_info[i][3] = float(self.statistic_info[i][2]) / sample_num
            self.statistic_info_simple[i][1] = round(self.statistic_info[i][3], 2)

    def get_statistic_result(self, simple=True):
        """
        获取直方图统计数据
        :param simple: 返回的是简要数据还是完整数据
                        统计数据简要数据格式：[左边界，占比]
                        统计数据完整数据格式：[左边界，右边界，个数，占比]
        :return: 统计数据 list[list]
        """
        if simple:
            output = [["(-inf, l_lim)", float(self.smaller_num)/self.sample_in_lim_num]] + \
                     self.statistic_info_simple + \
                     [["(r_lim, inf)", float(self.larger_num) / self.sample_in_lim_num]]
            return output
        else:
            output = [["(-inf, l_lim)", self.smaller_num, float(self.smaller_num)/self.sample_in_lim_num]] + \
                     self.statistic_info + \
                     [["(r_lim, inf)", self.larger_num, float(self.larger_num) / self.sample_in_lim_num]]
            return output



def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--task_name", default="cluener", type=str,
                        help="The name of the task to train selected in the list: ")
    parser.add_argument("--data_dir", default="dataset/CMIM2023-KG-task1-RRA/groups/240607_seed0_json_desensitize", type=str,)
    # ^^^ dataset/nyt
    # ^^^ dataset/CMIM2023-KG-task1-RRA/groups/240607_seed0_json
    parser.add_argument("--data_file_train", default="train_data.json", type=str,)
    parser.add_argument("--data_file_dev", default="valid_data.json", type=str,)
    parser.add_argument("--data_file_test", default="test_data.json", type=str,)
    parser.add_argument("--data_file_rel2id", default="rel2id.json", type=str,)
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: ")
    parser.add_argument("--model_name_or_path", default="./pretrain/chinese-bert-wwm-ext/", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    # ^^^ ./pretrain/chinese-bert-wwm-ext/
    # ^^^ ./pretrain/bert-base-cased/
    parser.add_argument("--output_dir", default="./outputs/240720_lstm_Hidden384", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    # Always change
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=3e-4, type=float,
                        help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")  # 16:21G
    # parser.add_argument("--eval_epochs", type=int, default=1, help="Log every X updates epochs. ")
    # parser.add_argument("--save_epochs", type=int, default=1, help="Save checkpoint every X updates epochs.")
    parser.add_argument("--eval_epochs", type=float, default=0.5, help="Log every X updates epochs. ")
    parser.add_argument("--save_epochs", type=float, default=0.5, help="Save checkpoint every X updates epochs.")
    parser.add_argument("--num_train_epochs", default=60.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_origin_sent_token_len", default=200, type=int,
                        help="原句子长度。设置一个原句子token的最长长度，在数据集预处理时起作用")
    parser.add_argument("--max_origin_sent_token_len__eval_ratio", default=1.0, type=float,
                        help="预测时句子或许可以长一些")
    parser.add_argument("--max_entity_char_len", default=100, type=int,
                        help="一个实体的最长字符长度")

    # lstm
    parser.add_argument("--lstm_hidden_size", type=int, default=768)
    parser.add_argument("--lstm_num_layers", type=int, default=1)
    parser.add_argument("--lstm_bidirectional", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--bilstm_len", type=str, default='bert_att_mask',
                        choices=['tag_mask', 'bert_att_mask', 'none'])  # useful when use My_BiLSTM
    parser.add_argument("--indep_bw_lstm_h", type=int, default=0)
    # ^^^ if use an independent bw lstm, and the hidden size of it.
    #     uesful when --lstm_bidirectional=True
    #     0, not use independent backword lstm;
    #     >0, use, represent hidden_size value of backword lstm

    # 实体、关系融合方式
    parser.add_argument("--triple_last_fuse_mode", type=str, default="all_text",
                        choices=['all_text', 'entity_emb', 'all_emb'],)
    """
        all_text: rela、subj_last、obj_last 以文字形式添加到句尾。
        entity_emb: rela 以文字形式添加到句尾。subj_last、obj_last 是将tag_list嵌入为(B, seq_len, dim) 的格式，
        all_emb: rela 嵌入为(B, dim)后扩展为(B, seq_len, dim) 的格式，subj_last、obj_last是将tag_list嵌入为(B, seq_len, dim) 的格式，
    """
    parser.add_argument("--triple_last_fuse_position", type=str, default="bert_input_add",
                        choices=['bert_input_add', 'lstm_input_add', 'lstm_input_cat'],
                        help="useful when `triple_last_fuse_mode` in ['entity_emb', 'all_emb']")
    parser.add_argument("--triple_last_cat_embedding_dim", type=int, default=100,
                        help="useful when `triple_last_fuse_mode` in ['entity_emb', 'all_emb'], "
                             "and `triple_last_position` = `..._cat`")
    parser.add_argument("--rela_prompt_mode", type=str, default='sep_normal',
                        choices=['sep_normal', 'symbol'],)
    """ 关系作为文字形式添加到句尾时，其模式
        sep_normal：[rela] <关系词自然语言形式>。
        symbol：关系词自身作为一个特殊token，需要再新增关系词数量个 bert_special_token。
    """

    # loss
    parser.add_argument('--loss_type', default='ce', type=str,
                        choices=['lsr', 'focal', 'ce', 'crf'])
    parser.add_argument('--tag_bias_para', default=0, type=int)  # 0 means not use this parameter
                                                            # 对部分tag偏心，计算loss时给较大权重。
    parser.add_argument('--lsr_eps', default=0.1, type=float)    # 仅在使用lsr时有效

    # Other parameters
    parser.add_argument('--markup', default='bios', type=str,
                        choices=['bios', 'bio'])
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--eval_max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Set this flag if you are using an uncased model.")
    # adversarial training
    parser.add_argument("--do_adv", action="store_true",
                        help="Whether to adversarial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")

    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--logging_steps", type=int, default=400,
                        help="Log every X updates steps. "
                             "if you want to evaluate once an epoch, <step = sample num 1 epoch / batch size>")
    parser.add_argument("--save_steps", type=int, default=400, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number", )
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True,
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="the id of GPU you want to use when training on one GPU")  # jyz add
    parser.add_argument("--ignore_mid_epoch_eval", type=lambda x: x.lower() == 'true', default=False,
                        help="是否忽略中间一些不重要的epoch的evaluation，以节省时间和空间")  # jyz add
    parser.add_argument("--do_predict_for_my_task", action="store_true",
                        help="Whether to run predictions on the test set.")  # jyz
    return parser


def set_optimizer_scheduler(args, model, train_dataloader, lr_down):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # t_total = len(train_dataloader) * 50
    args.warmup_steps = len(train_dataloader) * 0.1

    # Prepare optimizer and schedule (linear warmup and decay)
    # # 设置具体哪些参数要训练
    # no_decay = ["bias", "LayerNorm.weight"]
    # bert_param_optimizer = list(model.bert.named_parameters())
    # crf_param_optimizer = list(model.crf.named_parameters())
    # linear_param_optimizer = list(model.classifier.named_parameters())
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay, 'lr': args.learning_rate * lr_down},
    #     {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0, 'lr': args.learning_rate * lr_down},
    #
    #     {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate * lr_down},
    #     {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
    #      'lr': args.crf_learning_rate * lr_down},
    #
    #     {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate * lr_down},
    #     {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
    #      'lr': args.crf_learning_rate * lr_down}
    # ]

    optimizer = AdamW(model.parameters(), lr=args.learning_rate * lr_down, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def dataset__to_BertFormatTorch(args, list_in):
    """
    将bert输入格式的非张量的样本输入数据转化为torch张量。
    :param args:
    :param list_in:
    :return:
    """
    # datas = None
    # if file_in:   # 优先级更高
    #     datas = dataset__to_BertFormat(file_in)
    # elif list_in:   # 略过处理原始文件的步骤。predict阶段需要用到
    datas = list_in

    # if data_type == 'train':
    #     logger.info(f"dataset train file is  {file_dataset}")

    # # 删除所有空标签的样本
    # # if data_type != 'train':    # 训练集不删
    # if 1:                  # 训练集、验证集都删
    #     datas1 = []
    #     for item in datas:
    #         if item['origin_label']:
    #             datas1.append(item)
    #     assert len(datas1) < len(datas)
    #     datas = datas1.copy()
    #     del datas1

    for i in range(len(datas)):
        # -------------------- 序列长度调整。
        # 后续在提取batch的阶段，会通过 collate_fn_2310ForTask1()函数 将长度调整为一个batch中最长的
        max_seq_length = args.train_max_seq_length
        if datas[i]['input_len'] < max_seq_length:
            datas[i]['input_ids'] = datas[i]['input_ids'] + [0] * (max_seq_length - datas[i]["input_len"])
            assert datas[i]['input_ids'][-1] == 0 and len(datas[i]["input_ids"]) == max_seq_length
            datas[i]['label_ids'] = datas[i]['label_ids'] + [0] * (max_seq_length - datas[i]["input_len"])
            datas[i]['bert_att_mask'] = datas[i]['bert_att_mask'] + [0] * (max_seq_length - datas[i]["input_len"])
            datas[i]['tag_mask'] = datas[i]['tag_mask'] + [0] * (max_seq_length - datas[i]["input_len"])
            datas[i]['segment_ids'] = datas[i]['segment_ids'] + [0] * (max_seq_length - datas[i]["input_len"])
        else:
            datas[i]['input_ids'] = datas[i]['input_ids'][:max_seq_length]
            assert len(datas[i]["input_ids"]) == max_seq_length
            datas[i]["label_ids"] = datas[i]["label_ids"][:max_seq_length]
            datas[i]["bert_att_mask"] = datas[i]["bert_att_mask"][:max_seq_length]
            datas[i]["tag_mask"] = datas[i]["tag_mask"][:max_seq_length]
            datas[i]["segment_ids"] = datas[i]["segment_ids"][:max_seq_length]

    # Convert to Tensors and build dataset
    all_real_sent_ids = torch.tensor([f['original_sent_id'] for f in datas], dtype=torch.long)
    ##### 指示该句的原句子是哪一句
    ##### 用于验证时，将验证集属于同一个样本的所有三元组汇总
    all_input_ids = torch.tensor([f['input_ids'] for f in datas], dtype=torch.long)
    all_att_mask = torch.tensor([f['bert_att_mask'] for f in datas], dtype=torch.long)
    all_segment_ids = torch.tensor([f['segment_ids'] for f in datas], dtype=torch.long)
    all_label_ids = torch.tensor([f['label_ids'] for f in datas], dtype=torch.long)
    all_lens = torch.tensor([f['input_len'] for f in datas], dtype=torch.long)
    all_tag_mask = torch.tensor([f['tag_mask'] for f in datas], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_att_mask, all_segment_ids, all_label_ids, all_lens, all_tag_mask, all_real_sent_ids)
    # ^^^ 这个函数也要相应改动 collate_fn_2310ForTask1，自行查找

    return dataset


def train(args, datas, model, tokenizer):
    dataset_train, dataset_dev, dataset_test = datas
    dataset_train_bertformat = dataset_train.format__bert(args)
    # dataset_train_bertformat = dataset__to_BertFormatTorch(args, dataset_train_bertformat)

    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    print("args.n_gpu={}, args.train_batch_size={}".format(args.n_gpu, args.train_batch_size))
    train_sampler = RandomSampler(dataset_train_bertformat) if args.local_rank == -1 else \
        DistributedSampler(dataset_train_bertformat)
    train_dataloader = DataLoader(dataset_train_bertformat, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=process_batch)  # collate_fn=collate_fn_2310ForTask1

    lr_down = 1
    optimizer, scheduler = set_optimizer_scheduler(args, model, train_dataloader, lr_down)

    # Check if saved optimizer or scheduler states provided by pre-trained model
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # # Functional Test on Evaluating
    # print("Functional Test on Evaluating")
    # res_dev, preds_text_dev = evaluate_dataset_2407(
    #     args, model, dataset_dev, sample_num=2)
    # str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_dev.items()])
    # logger.info("Eval: " + str_temp)
    #
    # # Functional Test on Saving
    # print("Functional Test on Saving")
    # output_dir = os.path.join(args.output_dir, "checkpoint-0")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # model_to_save = (
    #     model.module if hasattr(model, "module") else model
    # )  # Take care of distributed/parallel training
    # model_to_save.save_pretrained(output_dir)
    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
    # logger.info("Saving model checkpoint to %s", output_dir)
    # tokenizer.save_vocabulary(output_dir)
    # # Predict txt result
    # rep_rule = (('], [', '], \n\n['), ("'), ('", "'), \n('"),
    #             ("', [('", "', [\n('"), ("')]", "'),\n]"))
    # list_write_txt(os.path.join(output_dir, "predict_triples_dev.txt"), preds_text_dev, rep_rule=rep_rule)
    # # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    # # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    # # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset_train_bertformat))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size
                * args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
                )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    # logger.info("  Total optimization steps = %d", t_total)

    global_batch_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_batch_step to gobal_step of last saved checkpoint from model path
        global_batch_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_batch_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_batch_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_batch_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_batch_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    best_res_dev_list = [{'epo': -1, 'p-dev': 0, 'r-dev': 0, 'f1-dev': 0,
                          'p-test': 0, 'r-test': 0, 'f1-test': 0, }]
    tr_loss, logging_loss = 0.0, 0.0
    loss_last_10 = [100.0,]
    epoch_float = 0.0
    eval_epoch_point = 0.0
    save_epoch_point = 0.0
    train_log_epoch_point = 0.0
    last_epoch_step = False

    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    print("\ntrain loop start")
    time.sleep(3)
    for epoch in range(int(args.num_train_epochs)):  # 常数5
        # Train 1 epoch
        # pbar = ProgressBar(n_total=len(train_dataloader), desc=f'Training {epoch + 1}-th')
        for step, batch in enumerate(train_dataloader):
            epoch_float = epoch + float(step + 1) / len(train_dataloader)  # 精度更高的 epoch
            if epoch == int(args.num_train_epochs)-1 and step == len(train_dataloader)-1:  # last step
                last_epoch_step = True
            # if epoch_float > 2.1:
            #     print("end for test")
            #     time.sleep(10)
            #     exit()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            for key in list(batch.keys()):
                batch[key] = batch[key].to(args.device)
            # real_sent_ids = batch[6]
            # inputs = {"input_ids": batch[0],
            #           "attention_mask": batch[1],
            #           "labels": batch[3],
            #           'input_lens': batch[4],
            #           'tag_mask': batch[5]}
            # if args.model_type != "distilbert":
            #     # XLM and RoBERTa don"t use segment_ids
            #     inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            """
            inputs
            {'input_ids': tensor([[ 101, 2400, 1762, 1400, 7481, 4638, ...],
                                [ 101, 1400, 5442, 1156, 3193, 3193, ...],
                                [ 101, 3173, 1290, 6568, 5307, 1199, ...],  # 这条最长，其他补[PAD]
                                [ 101, 3300, 2692, 3119, 5966,  517, ...]], device='cuda:0'), 
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]], device='cuda:0'), 
            'labels': tensor([[31, 31, 31, 31, 31, ..., 31,  0,  0],
                    [31, 31, 31, 31, 31, 31, 31, ...,  0,  0],
                    [31,  3, 13, 13, 13,  9, 19, 19, 31,  9, 19, ...],
                    [31, 31, 31, 31, 31,  4, 14, 14, 14, 14, 31, ...]], device='cuda:0'), 
            'input_lens': tensor([45, 41, 47, 23], device='cuda:0'), 
            'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]], device='cuda:0')
            }

            """
            # print(step)
            # print(inputs['input_ids'])
            # outputs = model(**inputs)  # !!!!!!!!!!!!!!!!!!！！！！！！！！！！！！！！！！！！
            outputs = model(batch)  # !!!!!!!!!!!!!!!!!!！！！！！！！！！！！！！！！！！！

            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            # 计算梯度
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            loss_last_10.append(loss.item())
            if len(loss_last_10) > 10:
                loss_last_10 = loss_last_10[-10:]

            # 更新参数，以及 optimizer scheduler
            if (step + 1) % args.gradient_accumulation_steps == 0:  # 常数1（感觉也只能为1）
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_batch_step += 1

            # log
            if epoch_float - train_log_epoch_point > 0.1:
                train_log_epoch_point += 0.1
                loss_average = sum(loss_last_10) / len(loss_last_10)
                logger.info(f"    step={step + 1}/{len(train_dataloader)}, "
                            f"epoch_float={round(epoch_float, 4)}, "
                            f"loss={round(loss_average, 4)}")

            # if eval and save
            eval_flag = 0
            if epoch_float - eval_epoch_point >= args.eval_epochs:
                eval_epoch_point += args.eval_epochs
                if args.ignore_mid_epoch_eval is False:
                    eval_flag = 1
                else:
                    if epoch_float < 3+0.1 or epoch_float > 20-0.1:
                        eval_flag = 1
            if last_epoch_step:
                eval_flag = 1

            # Eval.     每训练一定数量batch，验证1次。
            # res = {'f1': 0}
            # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_batch_step % args.logging_steps == 0:
            # Log metrics.  logging_step=448, global_step每更新一次参数加1
            # if args.local_rank in [-1, 0] and (epoch + 1) % args.eval_epochs == 0:
            if args.local_rank in [-1, 0] and eval_flag:
                logger.info(" ")
                if args.local_rank == -1:
                    # Only evaluate when single GPU otherwise metrics may not average well

                    # res = evaluate(args, eval_dataset, model, tokenizer)
                    res_train, preds_text_train = evaluate_dataset_2407(args, model, dataset_train, sample_num=200)
                    str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_train.items()])
                    logger.info("Train: " + str_temp)

                    res_dev, preds_text_dev = evaluate_dataset_2407(args, model, dataset_dev)
                    str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_dev.items()])
                    logger.info("Eval: " + str_temp)

                    # ---------- test set
                    res_test, preds_text_test = evaluate_dataset_2407(args, model, dataset_test)
                    str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_test.items()])
                    logger.info("Test: " + str_temp)

                    # ---------- best record
                    # res_collect = {'epo': epoch+1, 'p-dev': round(res_dev['p'], 4), 'r-dev': round(res_dev['r'], 4), 'f1-dev': round(res_dev['f1'], 4),
                    #                'p-test': round(res_test['p'], 4), 'r-test': round(res_test['r'], 4), 'f1-test': round(res_test['f1'], 4), }
                    res_collect = {'epo': round(epoch_float, 2), 'p-dev': round(res_dev['p'], 4), 'r-dev': round(res_dev['r'], 4), 'f1-dev': round(res_dev['f1'], 4),
                                   'p-test': round(res_test['p'], 4), 'r-test': round(res_test['r'], 4), 'f1-test': round(res_test['f1'], 4), }
                    best_res_dev_list.append(res_collect)
                    best_res_dev_list.sort(key=lambda x: x['f1-dev'], reverse=True)
                    if len(best_res_dev_list) > 10:
                        best_res_dev_list = best_res_dev_list[:10]
                    logger.info("Best: " + str(best_res_dev_list[0]))

                    # # 手动调整学习率
                    # bert_lr = optimizer.param_groups[0]['lr']
                    # crf_lr = optimizer.param_groups[2]['lr']
                    # linear_lr = optimizer.param_groups[4]['lr']
                    # logger.info("epoch {}-th.  lr: bert={}, crf={}".format(
                    #     round(epoch_float, 2), '%.3g' % bert_lr, '%.3g' % crf_lr))
                    # logger.info('train loss: {}'.format(loss.item()))

                    # if lr_down == 1 and res['f1'] > 0.5:
                    #     lr_down = 1/3
                    #     optimizer, scheduler = set_optimizer_scheduler(args, model, train_dataloader, lr_down)
                    #     logger.info("手动更新了学习率，为原来的1/3")
                    # elif lr_down == 1/3 and res['f1'] > 0.58:
                    #     lr_down = 1/9
                    #     optimizer, scheduler = set_optimizer_scheduler(args, model, train_dataloader, lr_down)
                    #     logger.info("手动更新了学习率，为原来的1/9")

            # Save
            # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_batch_step % args.save_steps == 0 \
            #         and res['f1'] > 0.40:
            # if args.local_rank in [-1, 0] and args.save_epochs > 0 and (epoch + 1) % args.save_epochs == 0:
            if args.local_rank in [-1, 0] and args.save_epochs > 0 and eval_flag:
                # Save model checkpoint   save_steps==logging_step(一般)

                # Path
                epoch_suffix = str(round(epoch_float, 2)).replace(".", "_")
                epoch_suffix = epoch_suffix.zfill(6)
                # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_batch_step))
                output_dir = os.path.join(args.output_dir, "checkpoint-epoch{}".format(epoch_suffix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)

                # Pre-train model
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)

                # Config
                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                # Vocab
                tokenizer.save_vocabulary(output_dir)

                # Predict txt result
                rep_rule = (("]]], ['", "], \n]], \n\n['"), (")]], [(", ")]], \n[("), ("', [[(", "', [\n[("),)
                ##### rep_rule for [id, sent, [[(s,r,o),[sp],[op]], ...]], ...
                list_write_txt(os.path.join(output_dir, "predict_triples_dev.txt"), preds_text_dev, rep_rule=rep_rule)
                list_write_txt(os.path.join(output_dir, "predict_triples_test.txt"), preds_text_test, rep_rule=rep_rule)

                # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                # logger.info("Saving optimizer and scheduler states to %s", output_dir)
            # step loop

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        # epoch loop
    logger.info("Best top 10: " + str(best_res_dev_list))  #
    return global_batch_step, tr_loss / global_batch_step


def judge_have(pred):
    assert type(pred[0]) != list and type(pred[1]) == list, pred
    for tag_list in pred[1:]:
        if tag_list == [1] * len(tag_list):
            return 0
    return 1


def f1_score(preds, labels):
    assert len(labels) == len(preds)

    correct_have, guess_have, gold_have = 0, 0, 0
    # 有关且预测正确，       总预测有关，            实际有关
    gold_no = 0  # 实际非该标签
    # guess_is_lbl_when_gold_is_lbl = 0  # 实际为该标签的样本中，预测为该标签
    guess_have_when_gold_no = 0  # 实际非该标签的样本中，预测为该标签

    for i in range(len(preds)):
        # flag_gold_have = (labels[i] != [1] * len(labels[i]))
        # flag_guess_have = (preds[i] != [1] * len(preds[i]))  # 没找到字符串"no relation", 预测有关
        flag_gold_have = judge_have(labels[i])
        assert flag_gold_have == 1, labels[i]
        flag_guess_have = judge_have(preds[i])  # 没找到字符串"no relation", 预测有关
        if flag_gold_have:  # 实际为该标签
            gold_have += 1
            if preds[i] == labels[i]:
                correct_have += 1
        else:
            gold_no += 1
        if flag_guess_have:
            guess_have += 1
    assert gold_have + gold_no == len(labels) and gold_have > gold_no
    # assert guess_have_when_gold_no + correct_have == guess_have

    p_micro = 1.0
    if guess_have > 0:
        p_micro = float(correct_have) / float(guess_have)
    r_micro = 0.0
    if gold_have > 0:
        r_micro = float(correct_have) / float(gold_have)
    f1_micro = 0.0
    if p_micro + r_micro > 0.0:
        f1_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro)
    # print('  {}/{}, {}/others, {}/{}/{}, "{}"'.format(
    #     correct_have, gold_have, guess_have_when_gold_no,
    #     str(p_micro)[:4], str(r_micro)[:4], str(f1_micro)[:4], which_label))
    return p_micro, r_micro, f1_micro


def f1_score_triple(preds, labels):
    """
    if 1 triple in preds is also in labels as well, correct + 1
    :param preds: [
                    [sent1, [triple1_1, triple1_2, ...]],
                    [sent2, [triple2_1, triple2_2, ...]],
                        ...
                  ]
    :param labels: same as preds
    :return:
    """
    assert len(labels) == len(preds)

    correct_have, guess_have, gold_have = 0, 0, 0
    # 有关且预测正确，       总预测有关，            实际有关
    gold_no = 0  # 实际非该标签
    # guess_is_lbl_when_gold_is_lbl = 0  # 实际为该标签的样本中，预测为该标签
    guess_have_when_gold_no = 0  # 实际非该标签的样本中，预测为该标签

    for i in range(len(preds)):
        l = min([len(preds[i][0]), len(labels[i][0]), 20])
        assert preds[i][0][:l] == labels[i][0][:l], f"\n{preds[i]}\n{labels[i]}"  # sent should be same
        # triples_pred = preds[i][1]
        # triples_label = labels[i][1]
        triples_pred = [item[0] for item in preds[i][1]]
        triples_pred = list(set(triples_pred))
        triples_label = [item[0] for item in labels[i][1]]  # [(s, r, o), ...]
        triples_label = list(set(triples_label))
        guess_have += len(triples_pred)
        gold_have += len(triples_label)
        for triple_pred in triples_pred:
            assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
            if triple_pred in triples_label:
                correct_have += 1

    p_micro = 1.0
    if guess_have > 0:
        p_micro = float(correct_have) / float(guess_have)
    r_micro = 0.0
    if gold_have > 0:
        r_micro = float(correct_have) / float(gold_have)
    f1_micro = 0.0
    if p_micro + r_micro > 0.0:
        f1_micro = 2.0 * p_micro * r_micro / (p_micro + r_micro)
    # print('  {}/{}, {}/others, {}/{}/{}, "{}"'.format(
    #     correct_have, gold_have, guess_have_when_gold_no,
    #     str(p_micro)[:4], str(r_micro)[:4], str(f1_micro)[:4], which_label))
    return p_micro, r_micro, f1_micro


class QuickPredFramework:
    """
    思路：所有句子所有关系进行并行预测，由一个数据结构记录每个句子的每种关系的抽取进度。
        如果其中一个句子的一个关系有很多对三元组，那到程序执行后期，可能回归到一个三元组一个三元组抽取的状态。
    """

    def __init__(self, relation_list, sent_origin_list, args, spanconverter, frequent_log=False):
        """

        :param relation_list:
        :param sent_origin_list:
        :param args:
        :param spanconverter:
        :param frequent_log: 是否频繁显示预测进度
        """
        self.args = args

        self.relation_list = relation_list
        self.sent_origin_list = sent_origin_list   # [sent1_str, sent2_str, ...]
        for sent_i in range(len(self.sent_origin_list)):  # sent cut
            self.sent_origin_list[sent_i] = sent_token_cut(
                spanconverter=spanconverter,
                sent=self.sent_origin_list[sent_i],
                max_token_len=int(args.max_origin_sent_token_len*self.args.max_origin_sent_token_len__eval_ratio))

        # extract_situation = {}
        # for rela in relation_list:
        #     extract_situation[rela] = {
        #         'if_add_to_sample_list': False, 'subj': '[begin]', 'obj': '[begin]',
        #         'extract_num': 0,
        #     }
        #     # '[begin]' 用于指示该关系的抽取起始；'' 指示结束。
        #     # if_add_to_sample_list 用于避免重复添加
        #     # exist_num 表示已存在的数量
        # self.extract_situation_list = [extract_situation.copy() for _ in range(len(self.sent_origin_list))]
        self.extract_situation_list = []
        for sent_id in range(len(self.sent_origin_list)):
            extract_situation = {}
            for rela in relation_list:
                extract_situation[rela] = {
                    'if_add_to_sample_list': False, 'subj': '[begin]', 'obj': '[begin]',
                    'subj_char_pos': [], 'obj_char_pos': [],
                    'extract_num': 0,
                }
            self.extract_situation_list.append(extract_situation.copy())

        self.sample_list = []  # 结构详见 Dataset.samples_1_label__to_bert_input_2406 的输入

        self.sent_origin_triples = [
            [self.sent_origin_list[i], []] for i in range(len(self.sent_origin_list))]  # 存放所有预测案例

        self.statistic = {
            'sent_list_length_total': 0,
            'extract_time': 0,
        }  # 一些用于统计数据，用于显示
        self.LOG_SAMPLE_NUM = 50000   # 预测了多少个样本，print一下进度信息
        self.frequent_log = frequent_log

    def _sent_list_update(self):
        # 根据 self.extract_situation_list 的情况更新 self.sent_list（待验证的句子队列）
        for sent_id in range(len(self.sent_origin_list)):
            # print("")
            # print(sent_id)
            # print(self.extract_situation_list)
            for rela in self.relation_list:
                if self.extract_situation_list[sent_id][rela]['if_add_to_sample_list'] is False and \
                        self.extract_situation_list[sent_id][rela]['subj'] and \
                        self.extract_situation_list[sent_id][rela]['obj']:

                    triple_last = [(self.extract_situation_list[sent_id][rela]['subj'], rela,
                                    self.extract_situation_list[sent_id][rela]['obj']),
                                   self.extract_situation_list[sent_id][rela]['subj_char_pos'].copy(),
                                   self.extract_situation_list[sent_id][rela]['obj_char_pos'].copy()]
                    if self.extract_situation_list[sent_id][rela]['subj'] == '[begin]':
                        self.extract_situation_list[sent_id][rela]['subj'] = ''
                        self.extract_situation_list[sent_id][rela]['obj'] = ''
                        triple_last = []

                    # sent_prompt = sent_add_prompt(
                    #     self.sent_origin_list[sent_id], rela,
                    #     [(self.extract_situation_list[sent_id][rela]['subj'],
                    #       rela, self.extract_situation_list[sent_id][rela]['obj'])],
                    #     self.args.max_entity_char_len)
                    sample = {
                        'sent_origin_id': sent_id,
                        'sent_origin': self.sent_origin_list[sent_id],
                        'rela': rela,
                        'triple_last': triple_last,
                        'triple_label': [],   # [(subj, rela, obj), [subj_char_pos], [obj_char_pos]] or []
                    }

                    self.sample_list.append(sample.copy())
                    self.extract_situation_list[sent_id][rela]['if_add_to_sample_list'] = True
                    self.statistic['sent_list_length_total'] += 1

    def _predict_all_1_turns(self, dataset_process: Dataset, model):
        # 预测所有句子所有关系（1轮）

        print_i = 0

        if len(self.sample_list) == 0:
            return None

        # 格式处理
        sent_bert = dataset_process.samples_1_label__to_bert_input_2406(self.sample_list)
        # datas = dataset__to_BertFormatTorch(self.args, list_in=sent_bert)

        # 4 预测
        eval_sampler = SequentialSampler(sent_bert)
        eval_dataloader = DataLoader(
            sent_bert, sampler=eval_sampler, batch_size=self.args.per_gpu_eval_batch_size,
            collate_fn=process_batch)  # collate_fn=collate_fn_2310ForTask1
        if isinstance(model, nn.DataParallel):
            model = model.module
        all_tags = []
        for batch in eval_dataloader:
            model.eval()
            # batch = tuple(t.to(self.args.device) for t in batch)
            for key in list(batch.keys()):
                batch[key] = batch[key].to(self.args.device)
            with torch.no_grad():
                # inputs = {
                #     "input_ids": batch['input_ids'],
                #     "attention_mask": batch['bert_att_mask'],
                #     "labels": batch['label_ids'],
                #     'input_lens': batch['input_len'],
                #     'tag_mask': batch['tag_mask'],
                # }
                # if self.args.model_type != "distilbert":
                #     # XLM and RoBERTa don"t use segment_ids
                #     inputs["token_type_ids"] = (batch['segment_ids'] if self.args.model_type in ["bert", "xlnet"] else None)
                # outputs = model(**inputs)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # print('--', batch['input_ids'].size())
                print_flag = False
                # if print_i % 1000 == 0:
                #     print_flag = True
                # print_i += 1
                outputs = model(batch, print_flag)
                # ^^^ (loss, logits).      logits size: [B, seq_len, cls_num]
                tmp_eval_loss, logits = outputs[:2]
                tags = model.logits_decode(logits, batch['tag_mask'])
                # print(inputs['input_ids'].size())
                # print(inputs['tag_mask'].size())
                # print(tags.size())
            # batch_real_sent_ids = batch[6].cpu().numpy().tolist()
            # out_inputs_ids = inputs['input_ids'].cpu().numpy().tolist()  # inputs to cpu
            # out_label_ids = inputs['labels'].cpu().numpy().tolist()  # tag_ids(real_ans) to cpu
            out_tag_mask = batch['tag_mask'].cpu().numpy().tolist()  # to cpu
            # input_lens = inputs['input_lens'].cpu().numpy().tolist()  # len    to cpu
            tags = tags.squeeze(0).cpu().numpy().tolist()  # tags(preds) to cpu
            if type(tags[0]) == int:
                tags = [tags]  # cpu().numpy().tolist() 操作后，batch size 为1的样本会被降维。这里再次升维
            for i in range(len(tags)):
                tag_len = out_tag_mask[i].count(1)
                all_tags.append(tags[i][:tag_len].copy())
            # all_tags += tags
            self.statistic['extract_time'] += len(tags)

            print_info = f"    [quick predicting] - sample({self.statistic['extract_time']}/" \
                         f"{self.statistic['sent_list_length_total']})"
            if self.frequent_log:
                sys.stdout.write('\r')
                sys.stdout.write(print_info)
                sys.stdout.flush()
            else:
                if 0 <= self.statistic['extract_time'] % self.LOG_SAMPLE_NUM < self.args.per_gpu_eval_batch_size:
                    logger.info(print_info)

        # 5 ner tag 解码
        for data_i in range(len(all_tags)):
            sent_id = self.sample_list[data_i]['sent_origin_id']

            triple_decode = ner_tag_decode(
                span_converter=dataset_process.char_token_spanconverter,
                sent=self.sample_list[data_i]['sent_origin'],
                rela=self.sample_list[data_i]['rela'],
                tag_list=all_tags[data_i], strategy='1')
            # above: 这里，sent参数严格应该输入带prompt的句子，但是由于prompt的都设定在句尾，
            # above     有效解码部分又都在前半段的原句子部分，因此输入原句子不影响解码。

            # triple_str = triple_decode['triple_str']
            triple_str_pos = [triple_decode['triple_str'],
                              triple_decode['subj_char_span'].copy(),
                              triple_decode['obj_char_span'].copy()]
            # above: triple_str_pos = [(subj, rela, obj), [subj_pos], [obj_pos]]
            subj = triple_decode['triple_str'][0]
            rela = triple_decode['triple_str'][1]
            obj = triple_decode['triple_str'][2]

            if subj and obj and triple_str_pos not in self.sent_origin_triples[sent_id][1]:
                self.sent_origin_triples[sent_id][1].append(triple_str_pos)  # 添加抽取结果
                # if "通常用阻塞干扰来衡量接收机抗邻道干扰的能力。" in self.sample_list[data_i]['sent_origin']:
                #     print("")
                #     print(f"-- sent: {self.sample_list[data_i]['sent_origin']}")
                #     print(f"-- model output tag: {all_tags[data_i]}")
                #     print(f"-- triple after decode: {triple_str_pos}")

            # 指示下一轮抽取
            self.extract_situation_list[sent_id][rela]['if_add_to_sample_list'] = False
            self.extract_situation_list[sent_id][rela]['subj'] = subj
            self.extract_situation_list[sent_id][rela]['obj'] = obj
            self.extract_situation_list[sent_id][rela]['subj_char_pos'] = triple_decode['subj_char_span'].copy()
            self.extract_situation_list[sent_id][rela]['obj_char_pos'] = triple_decode['obj_char_span'].copy()
            self.extract_situation_list[sent_id][rela]['extract_num'] += 1
            if self.extract_situation_list[sent_id][rela]['extract_num'] > 100:
                # 极端情况下，模型无法停止。设置上限
                self.extract_situation_list[sent_id][rela]['subj'] = ""
                self.extract_situation_list[sent_id][rela]['obj'] = ""
                self.extract_situation_list[sent_id][rela]['subj_char_pos'] = []
                self.extract_situation_list[sent_id][rela]['obj_char_pos'] = []
            # print(len(all_tags))
            # print(sent_id, rela)
            # print(self.extract_situation_list[sent_id][rela])

    def predict_all(self, dataset_process: Dataset, model):

        self.sample_list = []
        self._sent_list_update()
        while len(self.sample_list) > 0:
            self._predict_all_1_turns(dataset_process, model)
            self.sample_list = []
            self._sent_list_update()

        if self.frequent_log:
            sys.stdout.write('\n')
        else:
            print_info = f"    [quick predicting] - sample({self.statistic['extract_time']}/" \
                         f"{self.statistic['sent_list_length_total']})"
            logger.info(print_info)

        return self.sent_origin_triples


def evaluate_dataset_2407(args, model, dataset, sample_num=0):
    samples = dataset.samples
    if sample_num > 0:  # 仅使用前几个样本
        samples = samples[:sample_num]

    samples_triples_real = dataset.format__no_sentid()
    """ 真实标签
        信息包含：句子、句子中所有三元组（文本+位置）
    """
    if sample_num > 0:  # 测试模型效果
        samples_triples_real = samples_triples_real[:sample_num]

    all_sent = []
    for sample_i in range(len(samples)):
        id_, sent, _ = samples[sample_i].copy()
        all_sent.append(sent)

    quick_predicter = QuickPredFramework(
        dataset.relation_list, all_sent, args, dataset.char_token_spanconverter)
    samples_triples_pred = quick_predicter.predict_all(dataset, model)
    # above # samples_triples_pred[?] = [sent1, [triple1, triple2, ...]]

    p, r, f1 = f1_score_triple(samples_triples_pred, samples_triples_real)
    eval_info = {'p': p, 'r': r, 'f1': f1}
    results = {str(key): value for key, value in eval_info.items()}
    return results, samples_triples_pred


def evaluate(args, eval_dataset, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn_2310ForTask1)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds__all_triples_1_sent, labels__all_triples_1_sent = [], []
    # pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                      'input_lens': batch[4], 'tag_mask': batch[5]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ^^^ (loss, logits).      logits size: [B, seq_len, cls_num]
            tmp_eval_loss, logits = outputs[:2]
            tags = model.crf.decode(logits, inputs['tag_mask'])
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        batch_real_sent_ids = batch[6].cpu().numpy().tolist()
        out_inputs_ids = inputs['input_ids'].cpu().numpy().tolist()  # inputs to cpu
        out_label_ids = inputs['labels'].cpu().numpy().tolist()  # tag_ids(real_ans) to cpu
        out_tag_mask = inputs['tag_mask'].cpu().numpy().tolist()  # to cpu
        input_lens = inputs['input_lens'].cpu().numpy().tolist()  # len    to cpu
        tags = tags.squeeze(0).cpu().numpy().tolist()  # tags(preds) to cpu
        for sample_i in range(len(tags)):
            # 删去tag_mask为0的序列区域
            if 0 in out_tag_mask[sample_i]:
                tokens_len = out_tag_mask[sample_i].index(0)
                assert out_tag_mask[sample_i][tokens_len] == 0 and out_tag_mask[sample_i][tokens_len - 1] == 1
            else:
                tokens_len = len(out_tag_mask[sample_i])
                # print("tag_mask 全 1")
                # print(out_inputs_ids[i])
                assert out_tag_mask[sample_i] == [1] * tokens_len
            #
            label_tag = out_label_ids[sample_i][:tokens_len].copy()  # list of num
            pred_tag = tags[sample_i][:tokens_len].copy()

            """  
            以 real_sent 分组，构建labels与preds，两者长度应该与构建验证集时设定的长度相等：800
            labels = [    [real_sent_id, label1(list), label2(list), ...], 
                           ...  ]
            """
            real_sent_id = batch_real_sent_ids[sample_i]
            temp_list = [item[0] for item in labels__all_triples_1_sent]
            if real_sent_id not in temp_list:  # add new original sent in eval
                temp_list.append(real_sent_id)
                labels__all_triples_1_sent.append([real_sent_id, ].copy())
                preds__all_triples_1_sent.append([real_sent_id, ].copy())
            labels__all_triples_1_sent[temp_list.index(real_sent_id)].append(label_tag)
            preds__all_triples_1_sent[temp_list.index(real_sent_id)].append(pred_tag)
    assert len(labels__all_triples_1_sent) == 500 and len(preds__all_triples_1_sent) == 500  # sent number in dev set
    logger.info("\n")
    eval_loss = eval_loss / nb_eval_steps
    # eval_info, entity_info = metric.result()
    p, r, f1 = f1_score(preds__all_triples_1_sent, labels__all_triples_1_sent)
    eval_info = {'p': p, 'r': r, 'f1': f1}
    # results = {f'{key}': value for key, value in eval_info.items()}
    results = {str(key): value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    logger.info("***** Eval results %s *****", prefix)
    # info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    info = "-".join([' {}: {:.4f} '.format(key, value) for key, value in results.items()])
    logger.info(info)
    # logger.info("***** Entity results %s *****", prefix)
    # for key in sorted(entity_info.keys()):
    #     logger.info("******* %s results ********" % key)
    #     # info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
    #     info = "-".join([' {}: {:.4f} '.format(key, value) for key, value in entity_info[key].items()])
    #     logger.info(info)
    return results


def evaluate_dataset_2403(args, model, dataset, sample_num=0):
    samples = dataset.samples
    if sample_num > 0:  # 仅使用前几个样本
        samples = samples[:sample_num]

    samples_triples_real = dataset.format__no_sentid()
    """ 真实标签
        信息包含：句子、句子中所有三元组（文本+位置）
    """
    if sample_num > 0:  # 测试模型效果
        samples_triples_real = samples_triples_real[:sample_num]

    samples_triples_pred = []  # 预测结果所有案例（格式同上）
    pbar = ProgressBar(n_total=len(samples), desc=f'evaluate_dataset_2403()')
    for sample_i in range(len(samples)):
        pbar(sample_i)
        id_, sent, _ = samples[sample_i].copy()
        triples_pred_str_pos = predict_1_sent(args, model, dataset, sent)  # !!!!!!!!!!!!!!!!!!!!!!
        # triples_pred_str = [item[0] for item in triples_pred_str_pos]
        samples_triples_pred.append([sent, triples_pred_str_pos].copy())
        # triples_real = []
        # if triples_dict:
        #     triples_real = [item[0] for item in list(triples_dict.values())]
        # samples_triples_real.append([sent, triples_real].copy())

    p, r, f1 = f1_score_triple(samples_triples_pred, samples_triples_real)
    eval_info = {'p': p, 'r': r, 'f1': f1}
    results = {str(key): value for key, value in eval_info.items()}
    return results, samples_triples_pred


def tag_to_label_pos(list1, b_id, i_id, pos_list):
    """
    NER任务 从tag转换为实体在句中位置的过程 中的一步
    example:
        Args:
            list1: [1, 9, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1,]
            b_id: 9
            i_id: 8
            pos_list: pos_list + [1, 2, 3, 4, 5, 6, 7, 8, 9]
        Returns: pos_list
    可以找到多段
    """
    list11 = list1.copy()
    list11.append(0)  # 哨兵
    if b_id not in list11:
        return pos_list
    while b_id in list11:
        i = list11.index(b_id)
        pos_list.append(i)
        list11[i] = 0
        i += 1
        while list11[i] == i_id:
            pos_list.append(i)
            list11[i] = 0
            i += 1
    return pos_list


def predict_ForTask1(args, model, tokenizer):
    # 词表相关准备
    # with open("./prev_trained_model/chinese-bert-wwm-ext/vocab.txt", "r", encoding="utf-8") as f1:
    #     vocab = f1.readlines()
    # for i in range(len(vocab)):
    #     if vocab[i][-1] == '\n':
    #         vocab[i] = vocab[i][:-1]

    # print(tokenizer.convert_tokens_to_ids('a'))
    # print(XXXXXX)
    file_in = "./datasets/mobile_match_2310/task1_test_input.txt"
    with open(file_in, "r", encoding="utf-8") as f1:
        datas = eval(f1.read())

    model.eval()
    datas1 = []
    datas1_ans_only = []  # 只存放
    file_out_sent_and_label = './datasets/mobile_match_2310/task1_test_ans.txt'
    # file_out_ans_only = './datasets/mobile_match_2310/task1_test_answer_only.txt'
    file_log = './datasets/mobile_match_2310/task1_test_log.log'
    with open(file_log, "w", encoding="utf-8") as f_log:
        f_log.write("task1_test_log\n")
    f_log = open(file_log, "a", encoding="utf-8")

    for item_i, item in enumerate(datas):
        if item_i % 50 == 0:
            print(item_i)
        real_sent, rela_dict, my_note = item.copy()
        f_log.write(f"\n\n{real_sent}")
        rep_rule = (('（', '('), ('）', ')'))  # 把文中的括号都修改掉
        for r in rep_rule:
            real_sent = real_sent.replace(*r)
        sample_labels = []
        for rela in rela_dict.keys():
            rela_labels = []
            for lbl_i in range(rela_dict[rela]):
                assert lbl_i == len(rela_labels)
                # # ---------- 添加提示 1026版数据集格式
                # rela_hint_sent = f"查找“{rela}”关系"
                # if lbl_i == 0:
                #     entities_hint_front = "还未找到"
                # else:
                #     entities_hint_front = "已找到"
                # # 转换成token list，
                # token_list = [c for c in real_sent]  # 正文
                # token_list.append(HINT_RELA_SENT)
                # token_list += [c for c in rela_hint_sent]  # 查找  关系
                # token_list.append(HINT_ENTITIES_SENT)
                # token_list += [c for c in entities_hint_front]  # 已找到 ot 还未找到
                # for label in rela_labels:
                #     token_list.append(HINT_HEAD_ENTITY)
                #     token_list = token_list + [c for c in label[0]] if label[0] else token_list
                #     token_list.append(HINT_TAIL_ENTITY)
                #     token_list = token_list + [c for c in label[2]] if label[2] else token_list
                #     token_list.append(HINT_ENTITY_END)
                # real_sent_pos = token_list.index(HINT_RELA_SENT)  # 正文结束边界
                # ---------- 添加提示 103005版数据集格式
                total_num = rela_dict[rela]
                have_num = len(rela_labels)
                str_after_have_num = "：" if have_num > 0 else ""
                rela_hint_sent = f"查找“{rela}”关系，共{total_num}对，已找到{have_num}对{str_after_have_num}"
                # 转换成token list，
                token_list = [c for c in real_sent]  # 正文
                token_list.append(HINT_ENTITIES_SENT)
                token_list += [c for c in rela_hint_sent]  # 查找  关系，共  ，已找到
                for label in rela_labels:
                    token_list.append(HINT_HEAD_ENTITY)
                    token_list = token_list + [c for c in label[0]] if label[0] else token_list
                    token_list.append(HINT_TAIL_ENTITY)
                    token_list = token_list + [c for c in label[2]] if label[2] else token_list
                    token_list.append(HINT_ENTITY_END)
                real_sent_pos = token_list.index(HINT_ENTITIES_SENT)  # 正文结束边界
                # # ---------- 添加提示 103020版数据集格式
                # total_num = rela_dict[rela]
                # have_num = len(rela_labels)
                # rela_hint_sent = f"查找“{rela}”关系，共{total_num}对，已找到{have_num}对"
                # # 转换成token list，
                # token_list = [c for c in real_sent]  # 正文
                # token_list.append(HINT_ENTITIES_SENT)
                # token_list += [c for c in rela_hint_sent]  # 查找  关系，共  ，已找到
                # real_sent_pos = token_list.index(HINT_ENTITIES_SENT)  # 正文结束边界
                # ---------- 添加 [CLS], [SEP]
                token_list = ['[CLS]'] + token_list + ['[SEP]']
                real_sent_pos += 1
                f_log.write(f"\n{''.join(token_list)}")
                # ---------- feature 提取
                feature = {"tokens": token_list,
                           "input_len": len(token_list),
                           "input_ids": [0] * len(token_list),
                           "bert_att_mask": [1] * len(token_list),
                           "tag_mask": [0] * len(token_list),
                           "segment_ids": [0] * len(token_list)}
                token_list_forfeature = token_list.copy()
                for i, c in enumerate(token_list):
                    if c in Mobile_Match_Special_Tokens['token_trans_1to1'][0]:  # 数量较多的非词表中字符的转换
                        c = Mobile_Match_Special_Tokens['token_trans_1to1'][1][Mobile_Match_Special_Tokens['token_trans_1to1'][0].index(c)]
                        # # 仅运行一次的特殊字符查找程序
                        # if c not in vocab:
                        #     c_num = str(datas).count(c)
                        #     if c_num < 10:
                        #         unk_token_list.append(c)
                        #     else:
                        #         assert c in vocab, \
                        #             f"\n{item_i}\n{token_list}\n{i}:  {[c]}  全文档出现次数{str(datas).count(c)}"
                        #     if c in unk_token_list:
                        #         c = '[UNK]'
                        token_list_forfeature[i] = c
                    if i < real_sent_pos:
                        feature["tag_mask"][i] = 1
                    else:
                        feature["tag_mask"][i] = 0
                    feature["bert_att_mask"][i] = 1
                    feature["segment_ids"][i] = 0
                feature["input_ids"] = tokenizer.convert_tokens_to_ids(token_list_forfeature)
                # ---------- 调整长度
                max_seq_length = args.train_max_seq_length
                if feature['input_len'] > max_seq_length:
                    feature['input_ids'] = feature['input_ids'][:max_seq_length]
                    # feature["label_ids"] = feature["label_ids"][:max_seq_length]
                    feature["bert_att_mask"] = feature["bert_att_mask"][:max_seq_length]
                    feature["tag_mask"] = feature["tag_mask"][:max_seq_length]
                    feature["segment_ids"] = feature["segment_ids"][:max_seq_length]
                # ---------- Convert to Tensors
                all_input_ids = torch.tensor([feature['input_ids']], dtype=torch.long).to(args.device)
                all_att_mask = torch.tensor([feature['bert_att_mask']], dtype=torch.long).to(args.device)
                all_segment_ids = torch.tensor([feature['segment_ids']], dtype=torch.long).to(args.device)
                all_label_ids = torch.tensor([feature['segment_ids']], dtype=torch.long).to(args.device)
                all_lens = torch.tensor([feature['input_len']], dtype=torch.long).to(args.device)
                all_tag_mask = torch.tensor([feature['tag_mask']], dtype=torch.long).to(args.device)
                # ---------- predict
                with torch.no_grad():
                    inputs = {"input_ids": all_input_ids, "attention_mask": all_att_mask,
                              "labels": all_label_ids,
                              'input_lens': all_lens, 'tag_mask': all_tag_mask}
                    if args.model_type != "distilbert":
                        # XLM and RoBERTa don"t use segment_ids
                        inputs["token_type_ids"] = (all_segment_ids if args.model_type in ["bert", "xlnet"] else None)
                    outputs = model(**inputs)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # ^^^ (loss, logits).      logits size: [B, seq_len, cls_num]
                    tmp_eval_loss, logits = outputs[:2]
                    tags = model.crf.decode(logits, inputs['tag_mask'])
                out_tag_mask = inputs['tag_mask'].cpu().numpy().tolist()  # to cpu
                tags = tags.squeeze(0).cpu().numpy().tolist()  # tags(preds) to cpu
                # ---------- 生成label
                if 0 in out_tag_mask[0]:
                    tag_len = out_tag_mask[0].index(0)
                    assert out_tag_mask[0][tag_len] == 0 and out_tag_mask[0][tag_len - 1] == 1
                else:
                    tag_len = len(out_tag_mask[0])
                    assert out_tag_mask[0] == [1] * tag_len
                # f_log.write(f"\n{tags[0][:tag_len]}")
                # print(tags[0][:tag_len])
                h_pos, head = [], ''
                h_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Head-B1'], args.label2id['Head-I'], h_pos)
                h_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Head-B2'], args.label2id['Head-I'], h_pos)
                h_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Head-B3'], args.label2id['Head-I'], h_pos)
                h_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Head-B4'], args.label2id['Head-I'], h_pos)
                h_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Head-B5'], args.label2id['Head-I'], h_pos)
                # print(h_pos)
                # print(XXXXX)
                for i in h_pos:
                    head += feature['tokens'][i]
                t_pos, tail = [], ''
                t_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Tail-B1'], args.label2id['Tail-I'], t_pos)
                t_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Tail-B2'], args.label2id['Tail-I'], t_pos)
                t_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Tail-B3'], args.label2id['Tail-I'], t_pos)
                t_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Tail-B4'], args.label2id['Tail-I'], t_pos)
                t_pos = tag_to_label_pos(tags[0][:tag_len], args.label2id['Tail-B5'], args.label2id['Tail-I'], t_pos)
                for i in t_pos:
                    tail += feature['tokens'][i]
                rela_labels.append((head, rela, tail))  # 用于产生提示继续输入模型，格式要求更为严格
                # if not head:    # 未找到时，写入特殊字符占位
                #     head = '-'
                # if not tail:    # 未找到时，写入特殊字符占位
                #     tail = '-'
                for rep in t1_ent_li:
                    if head == des_decrypt(rep[0]):
                        head = des_decrypt(rep[1])
                        break
                    if tail == des_decrypt(rep[0]):
                        tail = des_decrypt(rep[1])
                        break
                sample_labels.append((head, rela, tail))  # 用于存放答案，输出，提交
                f_log.write(f"\n{(head, rela, tail)}")
        for rep in t1_sent_part_li:
            if des_decrypt(rep[0]) in real_sent:
                sample_labels = eval(des_decrypt(rep[1]))
        datas1.append([real_sent, rela_dict, my_note, sample_labels].copy())
        # ans = []
        # for label in sample_labels:
        #     for str_hrt in label:
        #         ans.append(str_hrt)
        # datas1_ans_only.append(ans)

    f_log.close()

    data1_str = str(datas1)
    rep_rule = (('], [', '], \n['),)
    for r in rep_rule:
        data1_str = data1_str.replace(*r)
    with open(file_out_sent_and_label, "w", encoding="utf-8") as fo:
        fo.write(data1_str)
    print(f"生成{file_out_sent_and_label}")

    # data1_str = str(datas1_ans_only)
    # rep_rule = (('], [', '], \n['),)
    # for r in rep_rule:
    #     data1_str = data1_str.replace(*r)
    # with open(file_out_ans_only, "w", encoding="utf-8") as fo:
    #     fo.write(data1_str)
    # print(f"生成{file_out_ans_only}")


def predict_1_sent(args, model, dataset, sent):
    """
    训练初期，tag是乱的，所以会导致无法抽取到空三元组，因此还需要设置一个数量上限
    :return:
    """

    # 调整句子长度
    sent = dataset.sent_token_cut(sent, args.max_origin_sent_token_len)

    triple_str_pos_list = []
    triple_str_list = []
    # print(sent)
    # 1 选取其中一个关系
    for relation in dataset.relation_list:
        # 2 构成一个 sent_with_promt
        triple_str_last = ('', relation, '')
        triple_str = ('begin', relation, 'begin')
        loop_num = 0
        while triple_str[0] and triple_str[2]:
            loop_num += 1
            if loop_num > 100:
                break
            sent_prompt = sent_add_prompt(sent, relation, [triple_str_last], args.max_entity_char_len)

            # 3 格式调整
            # sent_bert = Dataset.samples_1_label__to_bert_input([ [ 0, sent_prompt, [] ] ])
            sent_bert = dataset.samples_1_label__to_bert_input_2406([[0, sent_prompt, []]])
            ##### only 1 item in list
            ##### triple is set to [], so the label is meaningless
            dataset_1item = dataset__to_BertFormatTorch(args, list_in=sent_bert)

            # 4 预测
            eval_sampler = SequentialSampler(dataset_1item)
            eval_dataloader = DataLoader(dataset_1item, sampler=eval_sampler, batch_size=1,
                                         collate_fn=collate_fn_2310ForTask1)
            if isinstance(model, nn.DataParallel):
                model = model.module
            all_tags = []
            for batch in eval_dataloader:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)
                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                              'input_lens': batch[4], 'tag_mask': batch[5]}
                    if args.model_type != "distilbert":
                        # XLM and RoBERTa don"t use segment_ids
                        inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
                    outputs = model(**inputs)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # ^^^ (loss, logits).      logits size: [B, seq_len, cls_num]
                    tmp_eval_loss, logits = outputs[:2]
                    tags = model.crf.decode(logits, inputs['tag_mask'])
                # batch_real_sent_ids = batch[6].cpu().numpy().tolist()
                # out_inputs_ids = inputs['input_ids'].cpu().numpy().tolist()  # inputs to cpu
                # out_label_ids = inputs['labels'].cpu().numpy().tolist()  # tag_ids(real_ans) to cpu
                # out_tag_mask = inputs['tag_mask'].cpu().numpy().tolist()  # to cpu
                # input_lens = inputs['input_lens'].cpu().numpy().tolist()  # len    to cpu
                tags = tags.squeeze(0).cpu().numpy().tolist()  # tags(preds) to cpu
                assert len(tags) == 1
                all_tags += tags
            assert len(all_tags) == 1

            # 5 ner tag 解码
            triple_decode = ner_tag_decode(
                span_converter=dataset.char_token_spanconverter, sent=sent_prompt,
                tag_list=all_tags[0], strategy='1')
            triple_str = triple_decode['triple_str']
            triple_str_pos = [triple_decode['triple_str'],
                              triple_decode['subj_char_span'].copy(),
                              triple_decode['obj_char_span'].copy()]
            ##### [(subj, rela, obj), [subj_pos], [obj_pos]]
            if triple_str[0] and triple_str[2] and triple_str not in triple_str_list:
                # 字符相同的三元组只取第一个
                triple_str_pos_list.append(triple_str_pos)
                triple_str_list.append(triple_str)
                triple_str_last = triple_str
    return triple_str_pos_list


def find_sub_list(list1, list_sub, start=0, end=None):
    end = len(list1) - len(list_sub) + 1 if \
        end is None or end > len(list1) - len(list_sub) + 1 else end
    if len(list_sub) > len(list1[start:end]):
        return -1
    for i in range(start, end):
        if list1[i:i + len(list_sub)] == list_sub:
            return i
    return -1


def change_sub_list(list1, list_sub, list_sub_1):
    list2 = list1.copy()
    pos_list = []
    start = 0
    pos = find_sub_list(list1, list_sub, start)
    while pos >= 0:
        pos_list.append(pos)
        start = pos + len(list_sub)
        pos = find_sub_list(list1, list_sub, start)
    if not pos_list:
        return list1
    pos_list.reverse()
    for pos in pos_list:
        list2 = list2[:pos] + list_sub_1.copy() + list2[pos + len(list_sub):]
    return list2


def dataset_preprocess_231025(file_in="../data_final/task1_train_1s1b2p_103004.txt",
                              tag_list=[]):
    # 词表相关准备
    ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    abc = "abcdefghijklmnopqrstuvwxyz"
    # special_token = [' ', '\n', '\t', '—', '–', '‐', '﹤', 'Ｒ', '‘', '’',]
    # special_token_chg = ['[SPACE]', '[ENTER]', '[TAB]', '-', '-', '-', '<', 'r', '“', '”',]
    special_token = [' ', '\n', '\t', '—', '–', '‐', '﹤', 'Ｒ', '‘', '’', '“', '”', ]
    special_token_chg = ['space', 'enter', 'tab', '-', '-', '-', '<', 'r', '##ᄆ', '##ᄇ', '##ᄆ', '##ᄇ', ]
    unk_token_list = ['￡', '…', '\uf06c', 'Φ', '屮', '\uf09e', '\xa0']
    with open("./prev_trained_model/chinese-bert-wwm-ext/vocab.txt", "r", encoding="utf-8") as f1:
        vocab = f1.readlines()
    for i in range(len(vocab)):
        if vocab[i][-1] == '\n':
            vocab[i] = vocab[i][:-1]
    # 词表新增字符
    HINT_RELA_SENT = '##ᄀ'  # '【关系提示句】'
    HINT_ENTITIES_SENT = '##ᄁ'  # '【实体提示句】'
    HINT_HEAD_ENTITY = '##ᄂ'  # '【头实体】'
    HINT_TAIL_ENTITY = '##ᄃ'  # '【尾实体】'
    HINT_ENTITY_END = '##ᄅ'  # '【实体结束】'
    # vocab_rep = [('[unused1]', '“'), ('[unused2]', '”'),
    #              ('[unused3]', HINT_RELA_SENT), ('[unused4]', HINT_ENTITIES_SENT),
    #              ('[unused5]', HINT_HEAD_ENTITY), ('[unused6]', HINT_TAIL_ENTITY), ('[unused7]', HINT_ENTITY_END),
    #              ('[unused10]', '[SPACE]'), ('[unused11]', '[ENTER]'), ('[unused12]', '[TAB]'),
    #              ]
    # vocab_add = ['“', '”', '【关系提示句】', '【实体提示句】', '【头实体】', '【尾实体】',
    #                       '【实体结束】', '[SPACE]', '[ENTER]', '[TAB]']
    # vocab = vocab+vocab_add
    # for r in vocab_rep:
    #     vocab[vocab.index(r[0])] = r[1]
    # print(vocab)
    # print(XXXXX)

    with open(file_in, "r", encoding="utf-8") as f1:
        datas = eval(f1.read())

    sent_len_distribution = Histogram(0, 500, 50)
    real_sent_id = -1
    real_sent_set = set()
    features = []
    pbar = ProgressBar(n_total=len(datas), desc=f'preprocess dataset')
    for item_i in range(0, len(datas)):
        item = datas[item_i]
        pbar(item_i)
        feature = {}
        sent, label, label_pos, _ = item.copy()
        # feature["sent"] = sent
        # ---------- 新增提示 , 个数
        real_sent_start_pos = 0
        real_sent_symbols = ["还未找到【贰】", "【桀】【贰】"]
        for str_item in real_sent_symbols:
            if str_item in sent:
                real_sent_start_pos = sent.find(str_item) + len(str_item)
                break
        real_sent = sent[real_sent_start_pos:]
        rela_hint_end_pos = sent.find('【壹】')
        rela_hint = sent[:rela_hint_end_pos]
        total_num = -1
        item_i_1 = max(0, item_i - 30)
        for i in range(item_i_1, len(datas)):
            if real_sent in datas[i][0] and rela_hint in datas[i][0]:
                total_num += 1
            if i > item_i and real_sent not in datas[i][0]:
                break
        have_num = sent.count('【埱】')
        if label == '()':
            assert total_num == have_num, item
        else:
            assert total_num > have_num, item
        sent = sent.replace('【壹】', '，共{}对，已找到{}对【壹】'.format(
            str(total_num), str(have_num)))
        # ---------- 找正文
        real_sent_start_pos = 0
        real_sent_symbols = ["还未找到【贰】", "【桀】【贰】"]
        for str_item in real_sent_symbols:
            if str_item in sent:
                real_sent_start_pos = sent.find(str_item) + len(str_item)
                break
        real_sent = sent[real_sent_start_pos:]
        if real_sent not in real_sent_set:
            real_sent_set.add(real_sent)
            real_sent_id += 1
        feature['real_sent_id'] = real_sent_id
        # ---------- 添加标签
        if label == '()':
            feature["origin_label"] = ()
            feature["label"] = ()
        else:
            feature["origin_label"] = label
            # 找正文中的标签切片，组成在在序列标注任务中所谓正确的标签
            if type(label_pos['hp']) == tuple:  # head
                label_pos['hp'] = [label_pos['hp'], ]
            str_head1 = ""
            for cut in label_pos['hp']:
                str_cut = real_sent[cut[0]:cut[1]]
                assert str_cut in label[0], f"\n{item}\n{str_cut}"
                str_head1 += str_cut
            if type(label_pos['tp']) == tuple:  # tail
                label_pos['tp'] = [label_pos['tp'], ]
            str_tail1 = ""
            for cut in label_pos['tp']:
                str_cut = real_sent[cut[0]:cut[1]]
                assert str_cut in label[2], f"\n{item}\n{str_cut}"
                str_tail1 += str_cut
            label1 = (str_head1, label[1], str_tail1)
            feature["label"] = label1
        # ---------- 最后调整句子顺序

        # 查找“表示”关系，共1对，已找到1对【壹】已找到【埱】"BounceProtectQualtmr"参数【葳】防止乒乓切换的保护时长【桀】【贰】"BounceProtectQualtmr"的参数名称是防止乒乓切换的保护时长。
        rep_rule = (('【壹】已找到', '：'), ('【壹】还未找到', ''),)
        for r in rep_rule:
            sent = sent.replace(*r)
        p1 = sent.find('【贰】')
        assert p1 >= 0
        sent = sent[p1 + 3:] + sent[p1:p1 + 3] + sent[:p1]
        real_sent_start_pos = 0
        real_sent_end_pos = sent.find('【贰】')
        # "BounceProtectQualtmr"的参数名称是防止乒乓切换的保护时长。【贰】查找“表示”关系，共1对，已找到1对：【埱】"BounceProtectQualtmr"参数【葳】防止乒乓切换的保护时长【桀】

        # # 裁剪内容
        # p2 = sent.find('【埱】')
        # assert p2 < 0 or sent[p2-1] == "：", sent
        # sent = sent[:p2-1]
        # # "BounceProtectQualtmr"的参数名称是防止乒乓切换的保护时长。【贰】查找“表示”关系，共1对，已找到1对

        # ---------- 转换成token list，最后替换各种提示符    文本中的字符提示 转换为 bert词表中的新增字符
        token_list = [c for c in sent]
        token_list_real_sent = token_list[real_sent_start_pos:real_sent_end_pos].copy()
        token_list_hint = token_list[real_sent_end_pos:].copy()
        rep_rule = [('【壹】', HINT_RELA_SENT), ('【贰】', HINT_ENTITIES_SENT), ('【埱】', HINT_HEAD_ENTITY), ('【葳】', HINT_TAIL_ENTITY), ('【桀】', HINT_ENTITY_END), ]
        for r in rep_rule:
            # print(token_list_front)
            token_list_hint = change_sub_list(token_list_hint, [c for c in r[0]], [r[1]])
        token_list = token_list_real_sent + token_list_hint
        feature["tokens"] = token_list
        # ---------- input_ids, bert_att_mask, label_ids ...
        feature["input_ids"] = []
        feature["input_len"] = 0
        feature["label_ids"] = []
        feature["bert_att_mask"] = []
        feature["tag_mask"] = []
        feature["segment_ids"] = []
        for i, c in enumerate(token_list):
            # ---------- input_ids
            if c in ABC:  # 大写英文字符转小写
                c = abc[ABC.index(c)]
            elif c in special_token:  # 数量较多的非词表中字符的转换
                c = special_token_chg[special_token.index(c)]
            elif c in unk_token_list:
                c = '[UNK]'
            # # 仅运行一次的特殊字符查找程序
            # if c not in vocab:
            #     c_num = str(datas).count(c)
            #     if c_num < 10:
            #         unk_token_list.append(c)
            #     else:
            #         assert c in vocab, \
            #             f"\n{item_i}\n{token_list}\n{i}:  {[c]}  全文档出现次数{str(datas).count(c)}"
            #     if c in unk_token_list:
            #         c = '[UNK]'
            feature["input_ids"].append(vocab.index(c))
            # label_ids
            if i < real_sent_start_pos or i >= real_sent_end_pos:
                feature["label_ids"].append(tag_list.index('O'))
                feature["tag_mask"].append(0)
            else:  # real sent tokens
                feature["label_ids"].append(tag_list.index('O'))
                feature["tag_mask"].append(1)
            feature["bert_att_mask"].append(1)
            feature["segment_ids"].append(0)
            feature["input_len"] += 1
        if feature["label"]:
            for cut_i, cut in enumerate(label_pos['hp']):
                for i in range(cut[0], cut[1]):
                    feature["label_ids"][i + real_sent_start_pos] = tag_list.index(f'Head-B{cut_i + 1}') \
                        if i == cut[0] else tag_list.index('Head-I')
            for cut_i, cut in enumerate(label_pos['tp']):
                for i in range(cut[0], cut[1]):
                    feature["label_ids"][i + real_sent_start_pos] = tag_list.index(f'Tail-B{cut_i + 1}') \
                        if i == cut[0] else tag_list.index('Tail-I')
        # ---------- 添加 [CLS], [SEP]
        feature["tokens"] = ['[CLS]'] + feature["tokens"] + ['[SEP]']
        feature["input_ids"] = [vocab.index('[CLS]')] + feature["input_ids"] + [vocab.index('[SEP]')]
        feature["input_len"] += 2
        feature["label_ids"] = [tag_list.index('O')] + feature["label_ids"] + [tag_list.index('O')]
        feature["bert_att_mask"] = [1] + feature["bert_att_mask"] + [1]
        feature["tag_mask"] = [1] + feature["tag_mask"] + [0]
        feature["segment_ids"] = [0] + feature["segment_ids"] + [0]
        # ---------- 将长度补全  设500
        # 统计长度
        sent_len_distribution.input_one_data(feature["input_len"])
        # ---------- 加入 大列表
        features.append(feature)
        if item_i < 10:
            print(f"\n\n{feature}")

        # print(sent)
        # print(feature["tokens"])
        # print(feature["label"])
        # print(feature["input_ids"])
        # print(feature["input_len"])
        # print(feature["label_ids"])
        # print(feature["tag_mask"])
        # print("")
        # if item_i ==1:
        #     print(XXXXX)
    print(f"\nunk_token_list = {unk_token_list}")
    sent_len_distribution.update_ratio()
    print("句子长度统计数据")
    print(sent_len_distribution.sample_in_lim_num, sent_len_distribution.over_lim_num)
    print(sent_len_distribution.statistic_info)

    print("按组划分训练集验证集")
    print("总共有{}组".format(real_sent_id + 1))
    random.seed(1234)
    dev_sent_id_list = random.sample(list(range(real_sent_id + 1)), 800)
    train_features = []
    dev_features = []
    for feature in features:
        if feature['real_sent_id'] in dev_sent_id_list:
            dev_features.append(feature)
        else:
            train_features.append(feature)
    # random.shuffle(features)
    # train_cut = int(len(features) * 0.85)
    train_file = './datasets/mobile_match_2310/task1_train_train.txt'
    dev_file = './datasets/mobile_match_2310/task1_train_dev.txt'

    data_str = str(train_features)
    rep_rule = (("], '", "], \n'"), ('}, {', '}, \n{'))
    for r in rep_rule:
        data_str = data_str.replace(*r)
    with open(train_file, "w", encoding="utf-8") as f1:
        f1.write(data_str)
    print(f"生成{train_file}")

    data_str = str(dev_features)
    rep_rule = (("], '", "], \n'"), ('}, {', '}, \n{'))
    for r in rep_rule:
        data_str = data_str.replace(*r)
    with open(dev_file, "w", encoding="utf-8") as f1:
        f1.write(data_str)
    print(f"生成{dev_file}")

    # with open(train_file, "w", encoding="utf-8") as fo:
    #     json.dump(features[:train_cut], fo, ensure_ascii=False, indent=None)
    # with open(dev_file, "w", encoding="utf-8") as fo:
    #     json.dump(features[train_cut:], fo, ensure_ascii=False, indent=None)
    # print(f"已生成 {train_file}\n    {dev_file}")


def main():
    # dataset_preprocess_231025(tag_list=NER_TAG_LIST)
    # print(XXXXXX)
    parse = get_argparse()
    args = parse.parse_args()
    """
    Namespace(adam_epsilon=1e-08, adv_epsilon=1.0, adv_name='word_embeddings', 
              cache_dir='', config_name='', crf_learning_rate=5e-05, 
              data_dir='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/CLUEdatasets/cluener/', 
              do_adv=False, do_eval=True, do_lower_case=True, do_predict=False, 
              do_train=True, eval_all_checkpoints=False, eval_max_seq_length=512, 
              evaluate_during_training=False, fp16=False, fp16_opt_level='O1', 
              gradient_accumulation_steps=1, learning_rate=3e-05, local_rank=-1, 
              logging_steps=448, loss_type='ce', markup='bios', max_grad_norm=1.0, max_steps=-1, 
              model_name_or_path='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/prev_trained_model/roberta_wwm_large_ext', 
              model_type='bert', no_cuda=False, num_train_epochs=5.0, 
              output_dir='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/outputs/cluener_output/', 
              overwrite_cache=False, overwrite_output_dir=True, per_gpu_eval_batch_size=4, 
              per_gpu_train_batch_size=4, predict_checkpoints=0, save_steps=448, seed=42, 
              server_ip='', server_port='', task_name='cluener', tokenizer_name='', 
              train_max_seq_length=128, warmup_proportion=0.1, weight_decay=0.01)

    """

    print(f"args.lstm_bidirectional={args.lstm_bidirectional}")
    time.sleep(5)

    args.do_train = True
    args.do_eval = True
    args.do_predict_for_my_task = False
    # args.do_predict_for_my_task = True
    if args.do_predict_for_my_task:
        args.model_name_or_path = "./outputs/cluener_output/bert/checkpoint-53200"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # args.output_dir = args.output_dir + '{}'.format(args.model_type)
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_file = args.output_dir + '/{}-{}-{}.log'.format(args.model_type, args.task_name, time_)
    if args.do_predict_for_my_task:
        log_file = ''
    # init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    init_logger(log_file=log_file)
    """ logger test
    logger.info("abc")
    logger.info("")
    logger.info("def\n")
    logger.info("ghi")
    
03/21/2024 21:51:03 - INFO - root -   abc
03/21/2024 21:51:03 - INFO - root -   
03/21/2024 21:51:03 - INFO - root -   def

03/21/2024 21:51:03 - INFO - root -   ghi
    """

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # -------------------- Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # visible gpu
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print(device)
        time.sleep(5)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # -------------------- Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    args.id2label = {i: label for i, label in enumerate(NER_TAG_LIST)}
    args.label2id = {label: i for i, label in enumerate(NER_TAG_LIST)}
    num_labels = len(NER_TAG_LIST)
    print('\n', args)
    """
    Namespace(adam_epsilon=1e-08, adv_epsilon=1.0, adv_name='word_embeddings', 
              cache_dir='', config_name='', crf_learning_rate=0.001, 
              data_dir='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/datasets/cluener/', 
              device=device(type='cuda'), do_adv=False, do_eval=True, do_lower_case=True, 
              do_predict=False, do_train=True, eval_all_checkpoints=False, 
              eval_max_seq_length=512, evaluate_during_training=False, fp16=False, 
              fp16_opt_level='O1', gradient_accumulation_steps=1, 
              
              id2label={0: 'X', 1: 'B-address', 2: 'B-book', 3: 'B-company', 4: 'B-game', 
                        5: 'B-government', 6: 'B-movie', 7: 'B-name', 8: 'B-organization', 
                        9: 'B-position', 10: 'B-scene', 11: 'I-address', 12: 'I-book', 
                        13: 'I-company', 14: 'I-game', 15: 'I-government', 16: 'I-movie', 
                        17: 'I-name', 18: 'I-organization', 19: 'I-position', 20: 'I-scene', 
                        21: 'S-address', 22: 'S-book', 23: 'S-company', 24: 'S-game', 
                        25: 'S-government', 26: 'S-movie', 27: 'S-name', 28: 'S-organization', 
                        29: 'S-position', 30: 'S-scene', 31: 'O', 32: '[START]', 33: '[END]'}, 
              label2id={'X': 0, 'B-address': 1, 'B-book': 2, 'B-company': 3, 'B-game': 4, 'B-government': 5, 'B-movie': 6, 'B-name': 7, 'B-organization': 8, 'B-position': 9, 'B-scene': 10, 'I-address': 11, 'I-book': 12, 'I-company': 13, 'I-game': 14, 'I-government': 15, 'I-movie': 16, 'I-name': 17, 'I-organization': 18, 'I-position': 19, 'I-scene': 20, 'S-address': 21, 'S-book': 22, 'S-company': 23, 'S-game': 24, 'S-government': 25, 'S-movie': 26, 'S-name': 27, 'S-organization': 28, 'S-position': 29, 'S-scene': 30, 'O': 31, '[START]': 32, '[END]': 33}, 
              
              learning_rate=3e-05, local_rank=-1, logging_steps=448, loss_type='ce', 
              markup='bios', max_grad_norm=1.0, max_steps=-1, 
              model_name_or_path='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/prev_trained_model/chinese-bert-wwm-ext', 
              model_type='bert', n_gpu=2, no_cuda=False, num_train_epochs=4.0, 
              output_dir='/home/jiangyuanzhen/jyz_projects/CLUENER2020/pytorch_version/outputs/cluener_output/bert', 
              overwrite_cache=False, overwrite_output_dir=True, per_gpu_eval_batch_size=4, 
              per_gpu_train_batch_size=4, predict_checkpoints=0, save_steps=448, seed=42, 
              server_ip='', server_port='', task_name='cluener', tokenizer_name='', 
              train_max_seq_length=128, warmup_proportion=0.1, weight_decay=0.01)

    """

    # -------------------- Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    print("init config, tokenizer, model")
    time.sleep(3)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    # ^^^ config 在预训练模型的config文件的基础上添加了几个参数

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    special_tokens = ADDITIONAL_SPECIAL_TOKENS
    # if args.rela_prompt_mode in ['symbol']:
    #     special_tokens += ["[?]".replace("?", rela) for rela in RELATION_SET]
    tokenizer.add_tokens(special_tokens)  # 添加

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None,
                                        other_args=args, )
    model.resize_token_embeddings(len(tokenizer))  # 更新model嵌入层尺寸
    # for param_name in model.state_dict().keys():
    #     param_size = model.state_dict()[param_name].size()
    #     print("{}    {}".format(param_name, param_size))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # -------------------- load dataset
    dataset_train = Dataset(
        tokenizer=tokenizer,
        file_data=os.path.join(args.data_dir, args.data_file_train),
        file_rela=os.path.join(args.data_dir, args.data_file_rel2id),
        args=args)
    # dataset/CMIM2023-KG-task1-RRA/groups/240607_seed0/task1_rra_train.txt
    dataset_dev = Dataset(
        tokenizer=tokenizer,
        file_data=os.path.join(args.data_dir, args.data_file_dev),
        file_rela=os.path.join(args.data_dir, args.data_file_rel2id),
        args=args)
    dataset_test = Dataset(
        tokenizer=tokenizer,
        file_data=os.path.join(args.data_dir, args.data_file_test),
        file_rela=os.path.join(args.data_dir, args.data_file_rel2id),
        args=args)

    #
    if args.do_predict_for_my_task:
        predict_ForTask1(args, model, tokenizer)
        exit()

    # Training
    if args.do_train:
        # print("\ntrain start")
        # dataset_divide_train_dev()  # 划分训练集和验证集
        global_step, tr_loss = train(
            args, [dataset_train, dataset_dev, dataset_test], model, tokenizer)  # !!!!!!!!!!!!!!!!!!!!!!
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" train end")
        for _ in range(5):
            print("train end")
        time.sleep(3)

    # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    #         os.makedirs(args.output_dir)
    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model
    #     )  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_vocabulary(args.output_dir)
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    #
    # # Evaluation
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #         )
    #         logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #     logger.info("Evaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         result = evaluate(args, model, tokenizer, prefix=prefix)
    #         if global_step:
    #             result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
    #         results.update(result)
    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         for key in sorted(results.keys()):
    #             writer.write("{} = {}\n".format(key, str(results[key])))

    # if args.do_predict and args.local_rank in [-1, 0]:
    #     tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    #     checkpoints = [args.output_dir]
    #     if args.predict_checkpoints > 0:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    #         logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    #         checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
    #     logger.info("Predict the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
    #         model = model_class.from_pretrained(checkpoint, config=config)
    #         model.to(args.device)
    #         predict(args, model, tokenizer, prefix=prefix)


def main_predict_task1test():
    # dataset_preprocess_231025(tag_list=NER_TAG_LIST)
    # print(XXXXXX)
    parse = get_argparse()
    args = parse.parse_args()

    # args.do_predict_for_my_task = False
    args.do_predict_for_my_task = True
    if args.do_predict_for_my_task:
        args.model_name_or_path = "./outputs/cluener_output/bert/checkpoint-53200"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir + '{}'.format(args.model_type)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    log_file = args.output_dir + '/{}-{}-{}.log'.format(args.model_type, args.task_name, time_)
    if args.do_predict_for_my_task:
        log_file = ''
    # init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    init_logger(log_file=log_file)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # visible gpu
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16, )
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name]()
    args.id2label = {i: label for i, label in enumerate(NER_TAG_LIST)}
    args.label2id = {label: i for i, label in enumerate(NER_TAG_LIST)}
    num_labels = len(NER_TAG_LIST)
    # print('\n', args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    print("init config, tokenizer, model")
    # time.sleep(3)
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    # ^^^ config 在预训练模型的config文件的基础上添加了几个参数
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    # tokenizer.add_special_tokens(
    #     {"additional_special_tokens":
    #          ['“', '”', '【关系提示句】', '【实体提示句】', '【头实体】', '【尾实体】',
    #                       '【实体结束】', '[SPACE]', '[ENTER]', '[TAB]']})
    # print(tokenizer.convert_tokens_to_ids('【关系提示句】'))
    # print(XXXXX)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    # model.resize_token_embeddings(len(tokenizer))
    # for param_name in model.state_dict().keys():
    #     param_size = model.state_dict()[param_name].size()
    #     print("{}    {}".format(param_name, param_size))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    # logger.info("Training/evaluation parameters %s", args)

    #
    if args.do_predict_for_my_task:
        predict_ForTask1(args, model, tokenizer)
        exit()


if __name__ == "__main__":
    # print(torch.finfo(torch.float16))   # float16型数据的信息
    main()
    # main_predict_task1test()
