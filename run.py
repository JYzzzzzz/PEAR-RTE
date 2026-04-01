# import glob
# import logging
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
# from lib.callback.progressbar import ProgressBar
from lib.tools.common import seed_everything, list_write_txt
from lib.tools.common import init_logger, logger
from lib.models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from lib.models.bert_for_ner import BertCrfForNer, BertLstmForNer
from lib.models.albert_for_ner import AlbertCrfForNer
from lib.processors.utils_ner import CNerTokenizer, get_entities
# from lib.processors.ner_seq import convert_examples_to_features
from lib.processors.ner_seq import ner_processors as processors
# from lib.processors.ner_seq import collate_fn, collate_fn_2310ForTask1
# from lib.metrics.ner_metrics import SeqEntityScore

from dataset_loader import Dataset, NER_TAG_LIST, ADDITIONAL_SPECIAL_TOKENS, \
    sent_token_cut, ner_tag_decode, process_batch

from transformers import BertTokenizerFast


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertLstmForNer, BertTokenizerFast),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}


def get_argparse():
    parser = argparse.ArgumentParser()
    # Required parameters
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
    parser.add_argument("--checkpoint_dir_prefix", default="checkpoint-epoch", type=str,
                        help="the prefix of the name of folders in <output_dir>", )

    # Always change
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--crf_learning_rate", default=3e-4, type=float,
    #                     help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")  # 16:21G
    parser.add_argument("--eval_epochs", type=float, default=1.0, help="Log every X updates epochs. ")
    parser.add_argument("--save_epochs", type=float, default=1.0, help="Save checkpoint every X updates epochs.")
    parser.add_argument("--num_train_epochs", default=80.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_origin_sent_token_len", default=200, type=int,
                        help="max sentence length")
    parser.add_argument("--max_origin_sent_token_len__eval_ratio", default=1.0, type=float,
                        help="sentences could be longer when predicting.")
    parser.add_argument("--max_entity_char_len", default=100, type=int, help="")

    # lstm. useful in experiments, useless now
    parser.add_argument("--lstm_hidden_size", type=int, default=768)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    parser.add_argument("--lstm_bidirectional", type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--bilstm_len", type=str, default='bert_att_mask',
                        choices=['tag_mask', 'bert_att_mask', 'none'])  # useful when use My_BiLSTM
    parser.add_argument("--indep_bw_lstm_h", type=int, default=0)

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
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name", )
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    # parser.add_argument("--do_eval", action="store_true",
    #                     help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict_only", action="store_true",
                        help="Whether to run predictions on the test set.")  # jyz
    parser.add_argument("--do_lower_case", type=lambda x: x.lower() == 'true', default=True,
                        help="Set this flag if you are using an uncased model.")

    # adversarial training
    # parser.add_argument("--do_adv", action="store_true",
    #                     help="Whether to adversarial training.")
    # parser.add_argument('--adv_epsilon', default=1.0, type=float,
    #                     help="Epsilon for adversarial.")
    # parser.add_argument('--adv_name', default='word_embeddings', type=str,
    #                     help="name for adversarial layer.")

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
    parser.add_argument("--ignore_mid_epoch_eval", type=lambda x: x.lower() == 'true', default=False,
                        help="If ignore the evaluation of some unimportant epochs in the middle to save time and space")  # jyz add

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", type=bool, default=True,
                        help="Overwrite the content of the output directory")
    # parser.add_argument("--overwrite_cache", action="store_true",
    #                     help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html", )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    # parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="the id of GPU you want to use when training on one GPU")  # jyz add
    return parser


def set_optimizer_scheduler(args, model, train_dataloader, lr_down):
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # t_total = len(train_dataloader) * 50
    args.warmup_steps = len(train_dataloader) * args.warmup_proportion

    optimizer = AdamW(model.parameters(), lr=args.learning_rate * lr_down, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def train(args, datas, model, tokenizer):
    dataset_train, dataset_dev, dataset_test = datas
    dataset_train_bertformat = dataset_train.format__bert(args)

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
        for step, batch in enumerate(train_dataloader):
            epoch_float = epoch + float(step + 1) / len(train_dataloader)  # 精度更高的 epoch
            if epoch == int(args.num_train_epochs)-1 and step == len(train_dataloader)-1:  # last step
                last_epoch_step = True
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            for key in list(batch.keys()):
                batch[key] = batch[key].to(args.device)
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
                    if epoch_float < 3+0.1 or epoch_float > 30-0.1:
                        eval_flag = 1
            if last_epoch_step:
                eval_flag = 1

            # Eval.     每训练一定数量batch，验证1次。
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

                    # ---------- test set   jyz chg 2410
                    res_test, preds_text_test = evaluate_dataset_2407(args, model, dataset_test)
                    # str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_test.items()])
                    # logger.info("Test: " + str_temp)

                    # ---------- best record
                    # res_collect = {'epo': epoch+1, 'p-dev': round(res_dev['p'], 4), 'r-dev': round(res_dev['r'], 4), 'f1-dev': round(res_dev['f1'], 4),
                    #                'p-test': round(res_test['p'], 4), 'r-test': round(res_test['r'], 4), 'f1-test': round(res_test['f1'], 4), }
                    res_collect = {'epo': round(epoch_float, 2), 'p-dev': round(res_dev['p'], 4), 'r-dev': round(res_dev['r'], 4), 'f1-dev': round(res_dev['f1'], 4),
                                   'p-test': round(res_test['p'], 4), 'r-test': round(res_test['r'], 4), 'f1-test': round(res_test['f1'], 4), }
                    best_res_dev_list.append(res_collect)
                    best_res_dev_list.sort(key=lambda x: x['f1-dev'], reverse=True)
                    if len(best_res_dev_list) > 10:
                        best_res_dev_list = best_res_dev_list[:10]
                    # logger.info("Best: " + str(best_res_dev_list[0]))

            # Save
            if args.local_rank in [-1, 0] and args.save_epochs > 0 and eval_flag:

                # Path
                epoch_suffix = str(round(epoch_float, 2)).replace(".", "_")
                epoch_suffix = epoch_suffix.zfill(6)
                # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_batch_step))
                output_dir = os.path.join(
                    args.output_dir, f"{args.checkpoint_dir_prefix}{epoch_suffix}")
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

        logger.info("\n")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    # logger.info("Best top 10: " + str(best_res_dev_list))  #
    return global_batch_step, tr_loss / global_batch_step


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
                print_flag = False
                outputs = model(batch, print_flag)  # (loss, logits). logits size: [B, seq_len, cls_num]
                tmp_eval_loss, logits = outputs[:2]
                tags = model.logits_decode(logits, batch['tag_mask'])

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


def main():
    parse = get_argparse()
    args = parse.parse_args()

    if args.do_predict_only:
        # 从某文件中读取最优checkpoint
        with open(os.path.join(args.output_dir, "score_triple_complete.json"), 'r', encoding='UTF-8') as file1:
            data = json.loads(file1.read())
        best_checkpoint = data[-1]['best_checkpoint']
        temp = best_checkpoint.find(args.checkpoint_dir_prefix)
        best_checkpoint = best_checkpoint[temp:]
        args.model_name_or_path = os.path.join(args.output_dir, best_checkpoint)
        print(f"choose best_checkpoint: {args.model_name_or_path}")
        time.sleep(5)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    log_file = args.output_dir + '/{}-{}.log'.format(args.model_type, time_)
    if args.do_predict_only:
        log_file = ''
    init_logger(log_file=log_file)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

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
    args.id2label = {i: label for i, label in enumerate(NER_TAG_LIST)}
    args.label2id = {label: i for i, label in enumerate(NER_TAG_LIST)}
    num_labels = len(NER_TAG_LIST)

    # -------------------- Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    print("init config, tokenizer, model")
    time.sleep(3)
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    # ^^^ config 在预训练模型的config文件的基础上添加了几个参数

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None, )
    special_tokens = ADDITIONAL_SPECIAL_TOKENS
    if args.rela_prompt_mode in ['symbol']:  # use custom special tokens to represent relation types
        with open(os.path.join(args.data_dir, args.data_file_rel2id), "r", encoding="utf-8") as fi:
            relation_dict = eval(fi.read())
        relation_list = list(relation_dict.keys())
        special_tokens += ["[?]".replace("?", rela) for rela in relation_list]
    tokenizer.add_tokens(special_tokens)  # 添加
    print(f"special_tokens = {special_tokens}")
    print(f"len(tokenizer) = {len(tokenizer)}")

    model = model_class.from_pretrained(
        args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config,
        cache_dir=args.cache_dir if args.cache_dir else None, other_args=args, )
    for param_name in model.state_dict().keys():
        if "word_embeddings" in param_name:
            param_size = model.state_dict()[param_name].size()
            print("model size: {}  {}".format(param_name, param_size))
    time.sleep(5)
    model.resize_token_embeddings(len(tokenizer))
    # ^^^ 更新model嵌入层尺寸.
    #     从checkpoint读取模型时，word_embeddings的size已经是扩展后的了，上面这句可以删除，也可以不删。

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # -------------------- load dataset
    if not args.do_predict_only:
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

    # -------------------- predict
    if args.do_predict_only:
        print(f"do prediction ...")
        res_dev, preds_text_dev = evaluate_dataset_2407(args, model, dataset_dev)
        res_test, preds_text_test = evaluate_dataset_2407(args, model, dataset_test)
        # str_temp = ", ".join(['{}:{:.4f}'.format(key, value) for key, value in res_test.items()])
        # logger.info("Test: " + str_temp)
        print(f"\n--- !!! predict result: {res_test}\n")
        time.sleep(5)

        best_output_dir = os.path.join(args.output_dir, "best")
        best_output_dir = os.path.join(best_output_dir, best_checkpoint)
        if not os.path.exists(best_output_dir):
            os.makedirs(best_output_dir)

        rep_rule = (("]]], ['", "], \n]], \n\n['"), (")]], [(", ")]], \n[("), ("', [[(", "', [\n[("),)
        list_write_txt(os.path.join(best_output_dir, "predict_triples_dev.txt"), preds_text_dev, rep_rule=rep_rule)
        list_write_txt(os.path.join(best_output_dir, "predict_triples_test.txt"), preds_text_test, rep_rule=rep_rule)
        exit()

    # -------------------- training
    if args.do_train:
        global_step, tr_loss = train(
            args, [dataset_train, dataset_dev, dataset_test], model, tokenizer)  # !!!!!!!!!!!!!!!!!!!!!!
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info(" train end")
        for _ in range(5):
            print("train end")
        time.sleep(3)


if __name__ == "__main__":
    # print(torch.finfo(torch.float16))   # float16型数据的信息
    main()


