"""
更新说明：除了从label、pred中提取主客体那一部分。其他部分发生任何改动，version更新。

version: 241008
    -- 修改了默认的 pretrain model 路径
    -- 目前在llama-factory生成的qwen、glm的结果，都使用 evaluate_1_checkpoint__for_t5
    -- 添加了 evaluate_1_checkpoint__for_PRGC
    -- 修改了对 complex scenario 的统计框架，将它们放到了一个函数中
version: 240923
    -- 添加了 evaluate_1_checkpoint__for_t5
version: 240920
    -- 添加了大量部分统计的任务：--STATISTIC_RANGE=[triple_num, SEO, entity_len, ...]
version: 240918
    -- 修改 evaluate_1_checkpoint__for_SPN
    -- 添加 evaluate_1_checkpoint__for_UniRel
    -- 添加 evaluate_1_checkpoint__for_OneRel
version: 240901
    -- 写了 evaluate_1_checkpoint__for_TPLinker , 添加到 evaluate_1_checkpoint 选项中。
version: 240826
    -- 写了 evaluate_1_checkpoint__for_BiRTE , 添加到 evaluate_1_checkpoint 选项中。
    -- 减少了对本地其他文件的依赖
version: 240721
    -- 整合了 our model 中的 evaluate_1_checkpoint，将 evaluate_1_checkpoint 改为一个函数指针。
        因此目前该程序能直接用于 our model 与 SPN ，只需修改 evaluate_1_checkpoint
version: 240711
    -- 将大部分 配置参数 改写为可通过命令行输入的格式（argparse）
"""

import os
import time
import json
import yaml
import argparse
import matplotlib.pyplot as plt

import jieba
import rouge_chinese

""" ours
python get_score.py --OUTPUT_DIR="outputs/240903 Dataset Chg/240903_LSTM-nBiL1H768_LossLSR" --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6

"""

""" BiRTE
python3 get_score.py \
    --MODEL_NAME="BiRTE" \
    --PRETRAIN_MODEL_DIR="pretrained/chinese-bert-wwm-ext" \
    --DATASET_DIR="datasets/CMIM2023-NOM-task1-Re" \
    --LABEL_FILENAME_dev="dev.json" --LABEL_FILENAME_test="test.json"\
    --OUTPUT_DIR="outputs/240825_SeqLen200_lr3e-5/" \
    --PREDICT_FILENAME_dev="dev_pred.json" --PREDICT_FILENAME_test="test_pred.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \
    
"""

""" TPLinker
python3 get_score.py \
    --MODEL_NAME="tplinker" \
    --PRETRAIN_MODEL_DIR="models/chinese-bert-wwm-ext" \
    --DATASET_DIR="data4bert/CMIM2023-KG-task1-RRA/240607_seed0" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="tplinker/default_log_dir/240607" \
    --PREDICT_FILENAME_dev="prediction_valid.json" --PREDICT_FILENAME_test="prediction_test.json" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

存在一些额外参数，在 evaluate_1_checkpoint__for_TPLinker 函数开头。
"""

""" t5
python get_score.py \
    --MODEL_NAME="t5" \
    --PRETRAIN_MODEL_DIR="E:/JYZ_projects_python/J231014_MobileMatch/projects_for_paper/ner_code_231117/pretrain/chinese-bert-wwm-ext" \
    --DATASET_DIR="data/CMIM23-NOM1-RA" \
    --LABEL_FILENAME_dev="valid_data.json" --LABEL_FILENAME_test="test_data.json"\
    --OUTPUT_DIR="output/240921_Randeng77M_roseos" \
    --CHECKPOINT_FOLDER_PREFIX="checkpoint-step"
    --PREDICT_FILENAME_dev="dataset_prediction_dev_integ.json" --PREDICT_FILENAME_test="dataset_prediction_test_integ.json" \
    --llm_output_group="0,1,2,3,4,5,6,7,8,9" \
    --USE_ROUGE True --WHICH_ROUGE rouge-1 --ROUGE_THRE 0.6 \

"""



parser = argparse.ArgumentParser()

# ---------- 任务
parser.add_argument('--MODEL_NAME', type=str, default="ours",
                    choices=['ours', 'SPN', 'BiRTE', 'tplinker', 'UniRel', 'OneRel',
                             't5', 'PRGC'])
parser.add_argument('--STATISTIC_RANGE', type=str, default="all")
""" ^^^ STATISTIC_RANGE 统计范围
    -- all：对比所有样本所有三元组，输出 p，r，f1
    -- segmented_entity：对比所有样本中，含分段实体的三元组，p,r,f1结果有意义。
                    其中 p 以模型预测得到的含分段实体的三元组数量为分母；
    -- triple_num(?,??)：挑选三元组数量范围为 ? <= triple_num < ?? 的样本，p,r,f1结果有意义。
    -- EPO：统计 entity pair overlapping ，但是这种情况的样本量太少了，参考价值有限。
    -- SEO_sample: 挑选 含SingleEntityOverlapping 的样本，p,r,f1结果有意义。
    -- SEO_triple: 仅统计 SingleEntityOverlapping 的三元组， 仅r有意义。
    -- Normal
    -- sent_len(?,??): 挑选句子长度范围为 ? <= sent_len < ?? 的样本，p,r,f1结果有意义。
    -- entity_len(?,??): 仅统计 实体长度在一定范围内 的三元组， p,r,f1结果有意义。
    -- segment_num(?,??): 仅统计 实体段数在一定范围内 的三元组， 仅r有意义。
    -- pred_del_seg_ent: 删除ourmodel的预测的含 segmented entity 的三元组
    
    以上参数可进行组合，能进一步减小统计范围
"""

# ---------- 预测文件
parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/LSTM compare in LossCE/240725_LSTM-BiL2H576_LossCE")
# parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/nyt/nyt_LSTM-BiL2H576_LossCE")
parser.add_argument('--CHECKPOINT_FOLDER_PREFIX', type=str, default="checkpoint-epoch")  # 各checkpoint文件夹除数字部分外的前缀
parser.add_argument('--PREDICT_FILENAME_dev', type=str, default="predict_triples_dev.txt")
parser.add_argument('--PREDICT_FILENAME_test', type=str, default="predict_triples_test.txt")
parser.add_argument('--llm_output_group', type=str, default="0,1,2,3,4,5,6,7,8,9")
# ^^^ 大模型多组输出整合专用。当 MODEL_NAME in ['t5'] 时，生效

# ---------- 标签文件
parser.add_argument('--DATASET_DIR', type=str, default="dataset/CMIM23-NOM1-RA")
# parser.add_argument('--DATASET_DIR', type=str, default="dataset/nyt")
parser.add_argument('--LABEL_FILENAME_dev', type=str, default="valid_data.json")
parser.add_argument('--LABEL_FILENAME_test', type=str, default="test_data.json")

parser.add_argument('--PRETRAIN_MODEL_DIR', type=str, default="pretrain/chinese-bert-wwm-ext")
# parser.add_argument('--PRETRAIN_MODEL_DIR', type=str, default="pretrain/bert-base-cased")

# rouge
parser.add_argument('--USE_ROUGE', type=bool, default=False)
parser.add_argument('--WHICH_ROUGE', type=str, default="rouge-1")
parser.add_argument('--ROUGE_THRE', type=float, default=0.5)
parser.add_argument('--TOKEN_TYPE', type=str, default='tokenizer',
                    choices=['jieba', 'tokenizer'])

# # delete bad model
# parser.add_argument('--DELETE_BAD_MODEL', type=bool, default=False)  # 功能开关
# parser.add_argument('--CHECKPOINT_MODEL_NAME', type=str, default="pytorch_model.bin")
# parser.add_argument('--BAD_MODEL_THRE', type=float, default=0.5)  # 效果差的模型的f1阈值

args_global = parser.parse_args()

if args_global.OUTPUT_DIR[-1] == '\n':
    args_global.OUTPUT_DIR = args_global.OUTPUT_DIR[:-1]
if args_global.OUTPUT_DIR[-1] == '\r':   # 删除可能存在的回车换行符
    args_global.OUTPUT_DIR = args_global.OUTPUT_DIR[:-1]
LLM_Output_Group = args_global.llm_output_group.split(",")

# # tokenizer
# from transformers import BertTokenizer
# # from dataset_loader import ADDITIONAL_SPECIAL_TOKENS
# tokenizer_global = BertTokenizer.from_pretrained(args_global.PRETRAIN_MODEL_DIR, do_lower_case=True)
# # tokenizer_global = BertTokenizerFast.from_pretrained(
# #         PRETRAIN_MODEL_DIR, do_basic_tokenize=False,
# #         add_special_tokens=True, do_lower_case=True
# #     )
# ADDITIONAL_SPECIAL_TOKENS = ['“', '”']
# tokenizer_global.add_tokens(ADDITIONAL_SPECIAL_TOKENS)

# Relation_List = ['含有', '手段采用', '属性有', '组成部分', '功能', '实例为', '前提是', '特点', '定义', '影响', '别名', '造成', '分类']
# Definition, Alias, Instance, Characteristic,
# Contain, Attribute, Component, Category,
# Function, Approach, Premise, Cause, Influence
# 定义, 别名, 实例, 特点,    含有, 有属性, 组成部分, 分类,   功能, 实现手段, 前提, 造成, 影响
# [Contains', 'means used', 'attributes have', 'components',' functions', 'instances are', 'premises are', 'characteristics',' definitions', 'impacts',' aliases', 'causes',' classifications']

def process_after_BertTokenizer_decode(text_decode):  # jyz add 2024-07
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        run_args.model_dir, additional_special_tokens=added_token, do_basic_tokenize=False,
        add_special_tokens=True, do_lower_case=True)
    以上 tokenizer decode 时，汉字之间都会有空格，开头的##不会消除，因此进行手动处理
    """
    a2z = "abcdefghijklmnopqrstuvwxyz"
    text = ""
    # 手动去除token间的空格，但保留英文单词间的
    for i in range(len(text_decode)):
        if text_decode[i] == " ":
            if text_decode[i - 1] in a2z and text_decode[i + 1] in a2z:
                text += text_decode[i]
        else:
            text += text_decode[i]
    # 去除首尾的特殊字符
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.strip("#")
    # 将[UNK]改为一个字符
    text = text.replace("[UNK]", "?")
    return text


def suffix_find(input_str, symbol_l_str):
    """
    find the suffix sub-string after the last `symbol_l_str`
    """
    symbol_l_pos = input_str.find(symbol_l_str)  # find the position of left boundary symbol of span
    if symbol_l_pos < 0:
        return ""
    sub_pos_l = symbol_l_pos + len(symbol_l_str)

    sub_str = input_str[sub_pos_l:]

    symbol_l_pos2 = sub_str.find(symbol_l_str)
    while symbol_l_pos2 > -1:
        sub_pos_l += symbol_l_pos2 + len(symbol_l_str)
        sub_str = sub_str[symbol_l_pos2 + len(symbol_l_str):]
        symbol_l_pos2 = sub_str.find(symbol_l_str)

    return sub_str


def str_left_add(str_in: str, char: str, max_len: int):
    """
    在字符串左侧补 char，直到到达最大长度
    """
    while len(str_in) < max_len:
        str_in = char + str_in
    return str_in


def span_find(input_str, symbol_l_str, symbol_r_str, start=0, no_symbol_l_str_in_sub_str=True):
    """
    find the next sub-string between "span_l_str" and "span_r_str" in "input_str"
    version: 240921

    :param input_str:
    :param symbol_l_str: left boundary symbol of span
    :param symbol_r_str: right boundary symbol of span
    :param start: starting position for search
    :param no_symbol_l_str_in_sub_str: 是否要求子串中不含 `span_l_str`
        如果设为False。则 input_str="abcdab--yz", span_l_str="ab", span_r_str="yz" 时，
            sub_str="cdab--". 设为True时，sub_str="--"
    :return: (sub_string, sub_string_left_position, sub_string_right_positon)

    example:
        1. span_find("abc[123]defg[45]hijk", "[", "]", 0)
           return is ('123', 4, 7)
        2. span_find("abc[123]defg[45]hijk", "[", "]", 7)
           return is ('45', 13, 15)
        3. span_find("abc[123]defg[45]hijk", "[", "]", 15)
           return is ('', -1, -1)
        4. span_find("abc[123]defg[45]hijk", "[", "]", 13)
           return is ('', -1, -1)
    """

    symbol_l_pos = input_str.find(symbol_l_str, start)  # find the position of left boundary symbol of span
    if symbol_l_pos < 0:
        return "", -1, -1
    sub_pos_l = symbol_l_pos + len(symbol_l_str)

    symbol_r_pos = input_str.find(symbol_r_str, sub_pos_l)  # find the position of right boundary symbol of span
    if symbol_r_pos < 0:
        return "", -1, -1
    sub_pos_r = symbol_r_pos

    sub_str = input_str[sub_pos_l:sub_pos_r]

    symbol_l_pos2 = sub_str.find(symbol_l_str)
    while no_symbol_l_str_in_sub_str is True and symbol_l_pos2 > -1:
        # 截掉前缀，确保子串中没有span_l_str
        sub_pos_l += symbol_l_pos2 + len(symbol_l_str)
        sub_str = sub_str[symbol_l_pos2 + len(symbol_l_str):]
        symbol_l_pos2 = sub_str.find(symbol_l_str)

    return sub_str, sub_pos_l, sub_pos_r


def my_text_cut(text, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text, add_special_tokens=False))
    # for i in range(len(tokens) - 1, -1, -1):
    #     if tokens[i][0:2] == '##' and i > 0:
    #         tokens[i - 1] = tokens[i - 1] + tokens[i][2:]
    #         del tokens[i]
    return tokens  # a list


def span_have_overlap(span1, span2):
    # 判断是否有重叠
    # 情况1: span1的结束值大于span2的起始值，且span1的起始值小于span2的结束值
    # 情况2: span2的结束值大于span1的起始值，且span2的起始值小于span1的结束值
    # 两种情况之一为真，则存在重叠

    # 提取跨度的起始和结束值
    x1, y1 = span1
    x2, y2 = span2
    return (x1 < y2 and x2 < y1) or (x2 < y1 and x1 < y2)


def if_label_triple_num_in_range(triple_list: list, range1: tuple, ):
    """
    为限制统计范围而设计的函数。判断样本的三元组数量是否在范围内
    :param triple_list:
    :param range1:  (1,2) 切片规则
    :return:
    """
    if range1[0] <= len(triple_list) < range1[1]:
        return True
    else:
        return False


def if_sent_len_in_range(sent, tokenizer, range1: tuple, ):
    """
    为限制统计范围而设计的函数。
    判断 句子的长度（BertTokenizer生成的token数） 是否在范围内
    :param triple_list:
    :param range1:  (1,2) 切片规则
    :return:
    """
    tokens_id = tokenizer.encode(sent, add_special_tokens=False)
    if range1[0] <= len(tokens_id) < range1[1]:
        return True
    else:
        return False


def if_entity_len_in_range(entity_list, tokenizer, range1: tuple, ):
    """
    为限制统计范围而设计的函数。
    判断 句子的长度（BertTokenizer生成的token数） 是否在范围内
    :param range1:  (1,2) 切片规则
    :return:
    """
    entity_len = 0  # [tokenizer.encode(entity, add_special_tokens=False) for entity in entity_list]
    for entity in entity_list:
        entity_tokens = tokenizer.encode(entity, add_special_tokens=False)
        if len(entity_tokens) > entity_len:
            entity_len = len(entity_tokens)
    if range1[0] <= entity_len < range1[1]:
        return True
    else:
        return False


def if_segment_num_in_range(segment_num, range1: tuple):
    """
    为限制统计范围而设计的函数。
    判断 实体分段数 是否在范围内
    :param range1:  (1,2) 切片规则
    :return:
    """
    if range1[0] <= segment_num < range1[1]:
        return True
    else:
        return False



def if_sample_has_EPO(triple_list: list, ):
    """
    为限制统计范围而设计的函数。判断样本是否包含EntityPairOverlapping的情况。太少了
    :param triple_list:
    :return:
    """
    triple_ep_list = []  # 存放所有实体对
    triple_epo_list = []  # 存放epo情况的实体对
    for triple in triple_list:
        triple_ep = {triple['subject'], triple['object']}  # entity pair
        if triple_ep not in triple_ep_list:
            triple_ep_list.append(triple_ep)
        else:
            if triple_ep not in triple_epo_list:  # 添加到 triple_epo_list
                triple_epo_list.append(triple_ep)
    return triple_epo_list


def if_sample_has_SEO(triple_list):
    """
    为限制统计范围而设计的函数。判断样本是否包含SingleEntityOverlapping的情况。
    :param triple_list:
    :return:
    """
    entity_list = []
    enitiy_seo_set = set()
    for triple in triple_list:
        for entity in [triple['subject'], triple['object']]:
            if entity not in entity_list:
                entity_list.append(entity)
            else:
                enitiy_seo_set.add(entity)
    return enitiy_seo_set


def get_rouge(pred_txt: str, label_txt: str,
              args, tokenizer):  # 获取rouge分数
    if args.TOKEN_TYPE == 'jieba':
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
    elif args.TOKEN_TYPE == 'tokenizer':
        pred_tokens = my_text_cut(pred_txt, tokenizer)
        label_tokens = my_text_cut(label_txt, tokenizer)
    if len(pred_tokens) == 0:  # 为了防止 pred_tokens 为空
        pred_tokens = ['-+-+']
    assert len(label_tokens) > 0, f"\n{pred_txt}\n{label_txt}"

    rouge = rouge_chinese.Rouge()
    scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
    """scores = 
    [{
        'rouge-1': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}, 
        'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
        'rouge-l': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}
    }]
    example: 
        [a], [a b] --> rouge1: r=0.5, p=1, f1=0.67
        [a], [a b c] --> rouge1: r=0.33, p=1, f1=0.5(0.499...)
        [a b], [a c] --> rouge1: r=0.5, p=0.5, f1=0.5(0.499...)
    """
    return scores


def get_rouge_test(pred_txt: str, label_txt: str, tokenizer):  # 获取rouge分数
    pred_tokens = my_text_cut(pred_txt, tokenizer)
    print(f"pred_tokens = {pred_tokens}")
    label_tokens = my_text_cut(label_txt, tokenizer)
    print(f"label_tokens = {label_tokens}")
    if len(pred_tokens) == 0:  # 为了防止 pred_tokens 为空
        pred_tokens = ['-+-+']
    assert len(label_tokens) > 0, f"\n{pred_txt}\n{label_txt}"

    rouge = rouge_chinese.Rouge()
    scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
    print(scores)
    """scores = 
    [{
        'rouge-1': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}, 
        'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0}, 
        'rouge-l': {'r': 0.5, 'p': 1.0, 'f': 0.6666666622222223}
    }]
    example: 
        [a], [a b] --> rouge1: r=0.5, p=1, f1=0.67
        [a], [a b c] --> rouge1: r=0.33, p=1, f1=0.5(0.499...)
        [a b], [a c] --> rouge1: r=0.5, p=0.5, f1=0.5(0.499...)
    """
    return scores


def get_best_rouge_in_labels(triple_pred, triples_label_remain: list, args, tokenizer):
    """
        在考虑 rouge 的前提下，判断 triple_pred 是否存在与 triples_label_remain 中。
        1、在triples_label_remain中，查找与triple_pred最相似的triple_label
            最相似的标准为：关系必须相同，主体客体rouge中的较小值最大
        2、若最相似的rouge大于阈值，认为triple_pred存在于triples_label_remain
        3、若triple_pred存在于triples_label_remain，返回True，并删除triples_label_remain
            中的相应三元组并返回；若不存在，返回False，并返回triples_label_remain
    """
    assert len(triple_pred) == 3, f"\n{triple_pred}"

    best_label = {"triple_i": None, "rouge_score": 0}
    for triple_i in range(len(triples_label_remain)):
        triple_label = triples_label_remain[triple_i]
        # relation not same
        if triple_pred['rela'] != triple_label['rela']:
            continue
        # 获取 pred、label 主体间的 rouge
        subj_rouge_score = get_rouge(triple_pred['subj'], triple_label['subj'],
                                     args=args, tokenizer=tokenizer)[0]
        # 客体间的 rouge
        obj_rouge_score = get_rouge(triple_pred['obj'], triple_label['obj'],
                                    args=args, tokenizer=tokenizer)[0]
        # 融合rouge分数的策略1: 主客体 WHICH_ROUGE 的较小值
        triple_score = min(subj_rouge_score[args.WHICH_ROUGE]['f'],
                           obj_rouge_score[args.WHICH_ROUGE]['f'])
        # 更新best
        if triple_score > best_label['rouge_score']:
            best_label['rouge_score'] = triple_score
            best_label['triple_i'] = triple_i
            if best_label['rouge_score'] > 0.99:  # == 1.0
                break

    return best_label
    # if best_label['rouge_score'] > args.ROUGE_THRE:   # 阈值
    #     del triples_label_remain[best_label['triple_i']]
    #     return True
    # else:
    #     return False


def f1_score_triple(preds: list, labels: list, args, tokenizer):
    """
    if 1 triple in preds is also in labels as well, correct + 1
    :param preds: [
                    [triple1_1, triple1_2, ...],
                    [triple2_1, triple2_2, ...],
                        ...
                  ]
    :param labels: same as preds
    :return:
    """

    assert len(labels) == len(preds)
    assert type(labels[0]) == list
    assert type(preds[0]) == list, f"\n{preds[0]}"

    correct_have, guess_have, gold_have = 0, 0, 0
    # 有关且预测正确，       总预测有关，            实际有关
    gold_no = 0  # 实际非该标签
    # guess_is_lbl_when_gold_is_lbl = 0  # 实际为该标签的样本中，预测为该标签
    guess_have_when_gold_no = 0  # 实际非该标签的样本中，预测为该标签

    for i in range(len(preds)):

        # 获取每个样本的三元组
        triples_pred = preds[i]
        triples_label = labels[i]  # [(s, r, o), ...]

        guess_have += len(triples_pred)
        gold_have += len(triples_label)

        if args.USE_ROUGE is False:
            # 普通判别是否正确
            for triple_pred in triples_pred:
                assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
                if triple_pred in triples_label:
                    correct_have += 1
        else:
            # 添加rouge，判别是否正确
            triples_label_remain = triples_label.copy()
            for triple_pred in triples_pred:
                assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
                len_triples_label_remain = len(triples_label_remain)
                best_label = get_best_rouge_in_labels(triple_pred, triples_label_remain,
                                                      args=args, tokenizer=tokenizer)
                # if rouge_triple_in_labels(triple_pred, triples_label_remain, args=args):
                if best_label['rouge_score'] > args.ROUGE_THRE:
                    del triples_label_remain[best_label['triple_i']]
                    # assert len(triples_label_remain) == len_triples_label_remain - 1
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


def f1_score_sample(preds, labels):
    """
    if 1 triple in preds is also in labels as well, correct + 1
    :param preds: [
                    [triple1_1, triple1_2, ...],
                    [triple2_1, triple2_2, ...],
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

        triples_pred = preds[i]
        triples_label = labels[i]  # [(s, r, o), ...]

        if len(triples_pred) > 0:
            guess_have += 1
        if len(triples_label) > 0:
            gold_have += 1

        correct = 1
        for triple_pred in triples_pred:
            if triple_pred not in triples_label:
                correct = 0
                break
        for triple_label in triples_label:
            if triple_label not in triples_pred:
                correct = 0
                break
        correct_have += correct

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


def evaluate_1_checkpoint__for_ourmodel(predict_file, label_file, tokenizer):
    from dataset_loader import relation_modify

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = eval(f.read())
        """
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(len(predict_data), len(label_data))
    assert len(predict_data) == len(label_data)

    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    pred_data_i = 0
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- process label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                # triple_info['subj'] = subj_str,   # 实体取最后一个单词
                # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                triple_info['rela'] = relation_modify(rela_str, mode='webnlg')
                triple_info['obj'] = obj_str.split()[-1].lower()
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_with_pos = predict_data[d_i][1].copy()
        ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        triples_pred = []
        for triple_str_pos in triples_pred_with_pos:
            subj_str = triple_str_pos[0][0]
            rela_str = triple_str_pos[0][1]
            obj_str = triple_str_pos[0][2]
            subj_char_span = triple_str_pos[1]
            obj_char_span = triple_str_pos[2]

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                # triple_info['subj'] = subj_str,
                # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                # triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # 实体取最后一个单词
                triple_info['rela'] = relation_modify(rela_str, mode='webnlg')
                triple_info['obj'] = obj_str.split()[-1].lower()
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                triple_info1 = triple_info.copy()
                triple_info1['subj_char_span'] = subj_char_span.copy()
                triple_info1['obj_char_span'] = obj_char_span.copy()
                sample_out['triples_pred'].append(triple_info1)

        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f"triples_pred{data_i} = {triples_pred}")

        # triples_label_str = [tuple(triple[0]) for triple in triples_label]
        # triples_pred_str = [(triple['subject'], triple['predicate'], triple['object']) for triple in triples_pred_gather]
        # triples_pred_str = list(set(triples_pred_str))
        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    # input format:
    # print(all_samples_triples_pred[10])
    # print(all_samples_triples_label[10])
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_BiRTE(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """
            {
                "text": "系统消息4包含LAI、RACH控制参数信息。",
                "triple_list_pred": [
                    [ "系统消息4", "含有", "LAI控制参数信息" ],
                    [ "系统消息4", "含有", "RACH控制参数信息" ]
                ],
            },
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(len(predict_data), len(label_data))
    assert len(predict_data) == len(label_data)

    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    pred_data_i = 0
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- process label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {}
            if 'nyt' in args_global.DATASET_DIR:
                pass
                # triple_info['subj'] = subj_str.split()[-1].lower()   # 实体取最后一个单词
                # # triple_info['subj'] = subj_str,   # 实体取最后一个单词
                # # triple_info['rela'] = rela_str.replace('/', ' ').strip()
                # triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                # triple_info['obj'] = obj_str.split()[-1].lower()
                # triple_info['obj'] = obj_str,
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triple_list_pred'].copy()
        ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str[0]
            rela_str = triple_str[1]
            obj_str = triple_str[2]

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f"triples_pred{data_i} = {triples_pred}")

        # triples_label_str = [tuple(triple[0]) for triple in triples_label]
        # triples_pred_str = [(triple['subject'], triple['predicate'], triple['object']) for triple in triples_pred_gather]
        # triples_pred_str = list(set(triples_pred_str))
        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_SPN(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    # print(f"len(predict_data['gold']) = {len(predict_data['gold'])}")
    # print(f"len(predict_data['pred']) = {len(predict_data['pred'])}")
    # print(f"len(label_data) = {len(label_data)}")
    # assert len(predict_data['gold']) == len(label_data) and len(predict_data['gold']) == len(predict_data['pred'])
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # # -------------------- label
        # text_id = label_data[data_i]['id']
        # text = label_data[data_i]['text']
        # triples_label_with_pos = label_data[data_i]['relation_list_original'].copy()
        # ##### [[[subj,rela,obj], subj_pos, obj_pos], [...], ...]
        # # 处理标签实体
        # triples_label = []
        # for triple_str_pos in triples_label_with_pos:
        #     subj_str_origin = triple_str_pos[0][0]
        #     rela_str = triple_str_pos[0][1]
        #     obj_str_origin = triple_str_pos[0][2]
        #     # print(tokenizer_global.encode(subj_str_origin))
        #     # print(xxxxx)
        #     subj_str = tokenizer.decode(tokenizer.encode(subj_str_origin, add_special_tokens=False))
        #     subj_str = process_after_BertTokenizer_decode(subj_str)
        #     obj_str = tokenizer.decode(tokenizer.encode(obj_str_origin, add_special_tokens=False))
        #     obj_str = process_after_BertTokenizer_decode(obj_str)
        #     triples_label.append((subj_str, rela_str, obj_str))
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triple_pred_list'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str["subject"]
            rela_str = triple_str["predicate"]
            obj_str = triple_str["object"]

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)
        # triples_pred_0 = predict_data['pred'][str(data_i)].copy()
        # tokens_id = tokenizer.encode(text, add_special_tokens=True)
        # triples_pred = []
        # for triple_str_0 in triples_pred_0:
        #     """ example of triple_str_0
        #     [
        #         3, 0.999967098236084,
        #         1, 6, 0.9999997615814209, 1.0,
        #         24, 29, 0.9999912977218628, 0.9999896287918091
        #     ]
        #     """
        #     subj_tokens_id = tokens_id[triple_str_0[2]:triple_str_0[3] + 1]
        #     obj_tokens_id = tokens_id[triple_str_0[6]:triple_str_0[7] + 1]
        #     subj_str = tokenizer.decode(subj_tokens_id)
        #     subj_str = process_after_BertTokenizer_decode(subj_str)
        #     obj_str = tokenizer.decode(obj_tokens_id)
        #     obj_str = process_after_BertTokenizer_decode(obj_str)
        #     rela_str = relation_list[triple_str_0[0]]
        #     triple_str = (subj_str, rela_str, obj_str)
        #     if triple_str not in triples_pred:
        #         triples_pred.append(triple_str)
        #
        # if data_i < 3:
        #     print(f"triples_label{data_i} = {triples_label}")
        #     print(f" triples_pred{data_i} = {triples_pred}")
        #     print("")

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_TPLinker(predict_file, label_file, tokenizer):
    MAX_TOKEN_SEQ_LEN = 200
    TOKEN_SLIDING_LEN = 50
    tplinker_dataset_old_version = False  # 数据集格式版本为旧版

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
        """ format in tplinker output
            {
                "text": "系统消息4包含LAI、RACH控制参数信息。",
                "relation_list_pred": [
                    {
                        "subject": "ServingGW",
                        "object": "SAE-GW",
                        "subj_char_span": [
                            0,
                            9
                        ],
                        "obj_char_span": [
                            18,
                            24
                        ],
                        "predicate": "别名"
                    },
                    {...}, ...
                ]
            },
        """
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
        """
        """
    print(len(predict_data), len(label_data))

    # 由于tplinker中有split机制，过长的句子会被拆分为好几条，在此进行合并处理
    for sample_i in range(len(predict_data) - 1, -1, -1):
        if predict_data[sample_i]['tok_offset'] > 0:
            assert predict_data[sample_i]['id'] == predict_data[sample_i - 1]['id'], \
                f"\n{predict_data[sample_i - 1]['id']}\n{predict_data[sample_i]['id']}"

            # # idea1 全部合并到前一个，是否重复交给后续程序判断
            # predict_data[sample_i - 1]['relation_list_pred'] += predict_data[sample_i]['relation_list_pred'].copy()
            # del predict_data[sample_i]

            # idea2 丢弃 offset!=0 的样本（与idea1差不多，为了与其他模型近似，选择该策略）
            del predict_data[sample_i]

            # # idea3 offset!=0 的样本，只取新增范围内的三元组
            # new_sliding_span = (MAX_TOKEN_SEQ_LEN - TOKEN_SLIDING_LEN, 10000)  # 三元组获取范围
            # for triple_pred in predict_data[sample_i]['relation_list_pred']:
            #     if (span_have_overlap(new_sliding_span, triple_pred['subj_tok_span']) or
            #             span_have_overlap(new_sliding_span, triple_pred['obj_tok_span'])):
            #         triple_pred['subj_char_span'][0] += predict_data[sample_i]['char_offset']
            #         triple_pred['subj_char_span'][1] += predict_data[sample_i]['char_offset']
            #         triple_pred['obj_char_span'][0] += predict_data[sample_i]['char_offset']
            #         triple_pred['obj_char_span'][1] += predict_data[sample_i]['char_offset']
            #         triple_pred['subj_tok_span'][0] += predict_data[sample_i]['tok_offset']
            #         triple_pred['subj_tok_span'][1] += predict_data[sample_i]['tok_offset']
            #         triple_pred['obj_tok_span'][0] += predict_data[sample_i]['tok_offset']
            #         triple_pred['obj_tok_span'][1] += predict_data[sample_i]['tok_offset']
            #         predict_data[sample_i - 1]['relation_list_pred'].append(triple_pred.copy())
            # del predict_data[sample_i]

    # 由于tplinker中有split机制，过长的句子会被拆分为好几条，在此进行合并处理
    ########################################

    assert len(predict_data) == len(label_data), f"\n{len(predict_data)}\n{len(label_data)}"

    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    pred_data_i = 0
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- process label
        if tplinker_dataset_old_version:
            triples_label_with_pos = label_data[d_i]['relation_list_original'].copy()
        else:
            triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            if tplinker_dataset_old_version:
                subj_str = triple_str_pos[0][0]
                rela_str = triple_str_pos[0][1]
                obj_str = triple_str_pos[0][2]
                subj_char_span = triple_str_pos[1]
                obj_char_span = triple_str_pos[2]
            else:
                subj_str = triple_str_pos['subject']
                rela_str = triple_str_pos['predicate']
                obj_str = triple_str_pos['object']
                subj_char_span = triple_str_pos['subj_char_span']
                obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['relation_list_pred'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str["subject"]
            rela_str = triple_str["predicate"]
            obj_str = triple_str["object"]

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())
        """
        all_samples_triples_label = [
            [{'subj': ?, 'rela': ?, 'obj': ?}, {}, ...],
            ...
        ]
        """

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_UniRel(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triple_pred_list'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str = triple_str["subject"]
            rela_str = triple_str["predicate"]
            obj_str = triple_str["object"]

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_PRGC(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }

        # -------------------- label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triples_pred_list'].copy()
        triples_pred = []
        for triple_str in triples_pred_in:
            subj_str, rela_str, obj_str = triple_str.split("[sep]")

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_OneRel(predict_file, label_file, tokenizer):
    def process_after_tokenize_decode(str0):
        # 去除两侧残留的##，删除字符串中的空格
        str1 = str0.lstrip("##").replace(" ", "")
        return str1

    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
    assert len(predict_data) == len(label_data)

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # 遍历标签
        sample_out = {
            'text': label_data[d_i]['text'],
            # 'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_label': [],
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        # ^^^ { "subject": ?, "predicate": ?, "object": ?,
        #       "subj_char_span": ?, "obj_char_span": ?},
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            # 字符串处理，为了与有缺陷的prediction对齐
            subj_str = tokenizer.decode(tokenizer.encode(subj_str, add_special_tokens=False))
            subj_str = process_after_tokenize_decode(subj_str)
            subj_str = subj_str.replace('[UNK]', '?')
            obj_str = tokenizer.decode(tokenizer.encode(obj_str, add_special_tokens=False))
            obj_str = process_after_tokenize_decode(obj_str)
            obj_str = obj_str.replace('[UNK]', '?')

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)
                sample_out['triples_label'].append({
                    'subject': subj_str, 'predicate': rela_str, 'object': obj_str,
                    'subj_char_span': subj_char_span.copy(), 'obj_char_span': obj_char_span.copy()
                })

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triple_list_pred'].copy()
        triples_pred = []
        for triple in triples_pred_in:
            subj_str = triple[0]
            rela_str = triple[1]
            obj_str = triple[2]

            # 字符串处理
            subj_str = subj_str.replace('[UNK]', '?')
            obj_str = obj_str.replace('[UNK]', '?')

            # if "entity_len" in args_global.STATISTIC_RANGE:
            #     if if_entity_len_in_range(
            #             [subj_str, obj_str], tokenizer,
            #             eval(args_global.STATISTIC_RANGE[10:])) is False:
            #         continue

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_1_checkpoint__for_t5(predict_file, label_file, tokenizer):
    # 对一个checkpoint的一个train、dev、test中之一的数据进行验证打分
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
    assert len(predict_data) == len(label_data), f"\n{len(predict_data)}, {len(label_data)}"

    # -------------------- 遍历 label_data，获取结构化样本信息
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # 遍历标签
        text = label_data[d_i]['text']
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- label
        triples_label_with_pos = label_data[d_i]['relation_list'].copy()
        triples_label = []
        for triple_str_pos in triples_label_with_pos:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred_in = predict_data[d_i]['triple_pred_dict'].copy()
        """
            {"RLC分段[sep]功能[sep]与其他层的交互": "0, 1, 2, 3, 4, 5, 6", ...}
        """
        triples_pred = []
        for triple_str, appear_group_str in list(triples_pred_in.items()):
            subj_str, rela_str, obj_str = triple_str.split('[sep]')
            appear_group_list = appear_group_str.split(', ')
            appear_group_list = [ele for ele in appear_group_list
                                 if ele in LLM_Output_Group]
            # ^^^ 删除某些值的元素。
            if len(appear_group_list) * 2 <= len(LLM_Output_Group):
                continue    # 少于等于半数则判定为不正确

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)
                sample_out['triples_pred'].append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())
        all_samples_triples_pred_label_output.append(sample_out.copy())

    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    return res_triple, all_samples_triples_pred_label_output


def evaluate_complex_scenarios(samples_label_pred, tokenizer):
    """

    :param samples_label_pred:  list[dict]
        samples_label_pred[?] = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [{'subj': subj_str, 'rela': rela_str, 'obj': obj_str}, {}, ...],
        }

    :return:
    """

    if samples_label_pred is None or len(samples_label_pred) == 0:
        return
    if args_global.STATISTIC_RANGE == "all":  # 没有统计复杂场景的需求
        return

    all_samples_triples_label = []
    all_samples_triples_pred = []
    for sample in samples_label_pred:  # 遍历标签
        text = sample['text']

        # complex scenarios
        if "triple_num" in args_global.STATISTIC_RANGE:
            triple_value = eval(span_find(args_global.STATISTIC_RANGE, "triple_num", ")")[0] + ")")
            if if_label_triple_num_in_range(sample['triples_label'], triple_value) is False:
                continue
        # if "sent_len" in args_global.STATISTIC_RANGE:
        #     if if_sent_len_in_range(
        #             label_data[d_i]['text'], tokenizer,
        #             eval(args_global.STATISTIC_RANGE[8:])) is False:
        #         continue
        if 'EPO' in args_global.STATISTIC_RANGE:  # EPO situation, 太少了
            entity_epo_list = if_sample_has_EPO(sample['triples_label'])
            if len(entity_epo_list) == 0:
                continue
        if 'SEO' in args_global.STATISTIC_RANGE:  # SEO situation
            entity_seo_list = if_sample_has_SEO(sample['triples_label'])
            if len(entity_seo_list) == 0:
                continue
        if 'Normal' in args_global.STATISTIC_RANGE:  # situation without EPO and SEO
            entity_epo_list = if_sample_has_EPO(sample['triples_label'])
            entity_seo_list = if_sample_has_SEO(sample['triples_label'])
            if len(entity_epo_list) > 0 or len(entity_seo_list) > 0:
                continue

        # -------------------- label
        triples_label = []
        for triple_str_pos in sample['triples_label']:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            # complex scenarios
            if "segmented_entity" in args_global.STATISTIC_RANGE:
                # 删除 不含分段实体的三元组
                if len(subj_char_span) == 1 and len(obj_char_span) == 1:
                    continue
            if "segment_num" in args_global.STATISTIC_RANGE:
                segment_num = max([len(subj_char_span), len(obj_char_span)])
                range_value = eval(span_find(args_global.STATISTIC_RANGE, "segment_num", ")")[0] + ")")
                if if_segment_num_in_range(segment_num, range_value) is False:
                    continue
            if "SEO_triple" in args_global.STATISTIC_RANGE:  # 统计SEO情况的三元组的召回率
                if subj_str not in entity_seo_list and obj_str not in entity_seo_list:
                    continue
            if "entity_len" in args_global.STATISTIC_RANGE:
                value = eval(span_find(args_global.STATISTIC_RANGE, "entity_len", ")")[0] + ")")
                if if_entity_len_in_range([subj_str, obj_str], tokenizer, value) is False:
                    continue

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- process prediction
        triples_pred = []
        for triple_str in sample['triples_pred']:
            subj_str = triple_str['subj']
            rela_str = triple_str['rela']
            obj_str = triple_str['obj']

            # complex scenarios
            if "entity_len" in args_global.STATISTIC_RANGE:
                value = eval(span_find(args_global.STATISTIC_RANGE, "entity_len", ")")[0] + ")")
                if if_entity_len_in_range([subj_str, obj_str], tokenizer, value) is False:
                    continue
            if "segmented_entity" in args_global.STATISTIC_RANGE:
                # 删除 不含分段实体的三元组
                if 'subj_char_span' in triple_str:
                    if len(triple_str['subj_char_span']) == 1 and len(triple_str['obj_char_span']) == 1:
                        continue
                else:
                    if subj_str in text and obj_str in text:
                        continue
            if "pred_del_seg_ent" in args_global.STATISTIC_RANGE:
                # 删除预测结果中的所有含分段实体的三元组
                if 'subj_char_span' in triple_str:
                    if len(triple_str['subj_char_span']) > 1 or len(triple_str['obj_char_span']) > 1:
                        continue
                else:
                    if subj_str not in text or obj_str not in text:
                        continue

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_pred:
                triples_pred.append(triple_info)

        all_samples_triples_label.append(triples_label.copy())
        all_samples_triples_pred.append(triples_pred.copy())

    print(f"-- complex scenario: {args_global.STATISTIC_RANGE}")
    print(f"    sample_num={len(all_samples_triples_label)}")
    print(f"    label_triple_num={sum([len(triples) for triples in all_samples_triples_label])}, "
          f"pred_triple_num={sum([len(triples) for triples in all_samples_triples_pred])}")
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    print(f"    {res_triple}")


def score_all():
    # # pretrain model
    # config = AutoConfig.from_pretrained(args_from_yaml['model_name_or_path'])
    # print(f"pretain model config = {config}\n")
    #
    # tokenizer = T5Tokenizer.from_pretrained(args_from_yaml['model_name_or_path'], use_fast=False)
    # special_tokens_list = args_from_yaml['special_tokens']
    # if len(special_tokens_list) > 0:
    #     tokenizer.add_tokens(special_tokens_list)

    if args_global.MODEL_NAME == 'ours':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_ourmodel
    elif args_global.MODEL_NAME == 'BiRTE':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_BiRTE
    elif args_global.MODEL_NAME.lower() == 'tplinker':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_TPLinker
    elif args_global.MODEL_NAME == 'SPN':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_SPN
    elif args_global.MODEL_NAME == 'UniRel':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_UniRel
    elif args_global.MODEL_NAME == 'PRGC':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_PRGC
    elif args_global.MODEL_NAME == 'OneRel':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_OneRel
    elif args_global.MODEL_NAME == 't5':
        evaluate_1_checkpoint = evaluate_1_checkpoint__for_t5

    # tokenizer
    tokenizer = None
    if args_global.MODEL_NAME in ['OneRel']:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args_global.PRETRAIN_MODEL_DIR, do_lower_case=True)
    else:
        from transformers import BertTokenizer
        # from dataset_loader import ADDITIONAL_SPECIAL_TOKENS
        tokenizer = BertTokenizer.from_pretrained(args_global.PRETRAIN_MODEL_DIR, do_lower_case=True)
        ADDITIONAL_SPECIAL_TOKENS = ['“', '”']
        tokenizer.add_tokens(ADDITIONAL_SPECIAL_TOKENS)

    # get checkpoint dir
    checkpoint_dir_list = []
    folders = list(os.walk(args_global.OUTPUT_DIR))[0][1]
    for folder in folders:
        if args_global.CHECKPOINT_FOLDER_PREFIX in folder:
            checkpoint_dir_list.append(os.path.join(args_global.OUTPUT_DIR, folder))
    checkpoint_dir_list.sort(
        key=lambda x: str_left_add(suffix_find(x, args_global.CHECKPOINT_FOLDER_PREFIX), "0", 9))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"len of checkpoint_dir_list is {len(checkpoint_dir_list)}")
    time.sleep(5)

    res_all = []
    for checkpoint_dir in checkpoint_dir_list:
        res = {
            'step': checkpoint_dir,
        }
        print(f"\nstep={checkpoint_dir}")

        predict_file = os.path.join(checkpoint_dir, args_global.PREDICT_FILENAME_dev)
        label_file = os.path.join(args_global.DATASET_DIR, args_global.LABEL_FILENAME_dev)
        if os.path.isfile(predict_file) is True:
            res_triple_dev, samples_label_pred_dev = evaluate_1_checkpoint(
                predict_file=predict_file, label_file=label_file, tokenizer=tokenizer
            )
            res['triple_dev'] = res_triple_dev
            print(f"triple dev: {res_triple_dev}")
            evaluate_complex_scenarios(samples_label_pred_dev, tokenizer)

        predict_file = os.path.join(checkpoint_dir, args_global.PREDICT_FILENAME_test)
        label_file = os.path.join(args_global.DATASET_DIR, args_global.LABEL_FILENAME_test)
        if os.path.isfile(predict_file) is True:
            res_triple_test, samples_label_pred_test = evaluate_1_checkpoint(
                predict_file=predict_file, label_file=label_file, tokenizer=tokenizer,
            )
            res['triple_test'] = res_triple_test
            print(f"triple test: {res_triple_test}")
            evaluate_complex_scenarios(samples_label_pred_test, tokenizer)
            # evaluate_complex_scenarios(samples_label_pred_test+samples_label_pred_dev, tokenizer)  # for EPO
        # res_triple_test = evaluate_1_checkpoint_240424(os.path.join(output_dir, "dataset_prediction_test.json"), ask_num, tokenizer)
        # res['triple_test'] = res_triple_test
        # print(f"triple test: {res_triple_test}")

        if 'triple_dev' in res:  # and 'triple_test' in res:
            res_all.append(res)
        else:
            print("    该checkpoint无完整的案例输出，无法打分")

    res_all.sort(key=lambda x: x['triple_dev']['f1'], reverse=True)
    # res_all_TestTop = res_all.copy()
    # res_all_TestTop.sort(key=lambda x: x['triple_test']['f1'], reverse=True)  # 即，将dev、test角色颠倒

    # decide the best checkpoint
    best_checkpoint = res_all[0]['step']

    # # 计算 valid 与 test f1 之间差值的平均值
    # res_top10 = res_all[:10].copy() if len(res_all) > 10 else res_all.copy()
    # f1_dif = 0
    # for res in res_top10:
    #     f1_dif += res['triple_test']['f1'] - res['triple_dev']['f1']
    # f1_dif /= len(res_top10)
    # print(f"f1_dif_between_dev_test = {f1_dif}")

    res_all.append({
        # 'aver_top10_f1_dif': f1_dif,
        'best_checkpoint': best_checkpoint,  # for prediction
    })    # additional information

    # complex scenarios
    if args_global.STATISTIC_RANGE != "all":
        print("complex scenario static, no save and curve.")
        exit()  # do not save

    # save
    rouge_suffix = "complete"
    if args_global.USE_ROUGE:
        # rouge_suffix = f"{args_global.WHICH_ROUGE.replace('-', '')}" \
        #                f"({args_global.ROUGE_THRE},{args_global.TOKEN_TYPE})"
        rouge_suffix = f"{args_global.WHICH_ROUGE.replace('-', '')}" \
                       f"({args_global.ROUGE_THRE})"

    file_name = f"score_triple_{rouge_suffix}.json"
    file_name_TestTop = f"score_triple(TestTop)_{rouge_suffix}.json"

    with open(os.path.join(args_global.OUTPUT_DIR, file_name), "w", encoding="utf-8") as fp:
        json.dump(res_all, fp, ensure_ascii=False, indent=4)
    # with open(os.path.join(args_global.OUTPUT_DIR, file_name_TestTop), "w", encoding="utf-8") as fp:
    #     json.dump(res_all_TestTop, fp, ensure_ascii=False, indent=4)
    print(f"score_triple.json saved")
    return os.path.join(args_global.OUTPUT_DIR, file_name)


def paint_f1_curve(file_path_name):
    print("paint curve")

    file_path = os.path.dirname(file_path_name)
    file_name = os.path.basename(file_path_name)
    print("file_path: ", file_path)
    print("file_name: ", file_name)
    time.sleep(5)

    with open(file_path_name, "r", encoding="utf-8") as fp:
        res_all = json.load(fp)
    for i in range(len(res_all) - 1, -1, -1):
        if res_all[i].get('step') is None:
            del res_all[i]  # {'aver_top10_f1_dif': -0.0204}
    res_all.sort(key=lambda x: str_left_add(suffix_find(x['step'], args_global.CHECKPOINT_FOLDER_PREFIX), "0", 9))

    for res in res_all:
        print("  ", res['step'])
    print("check the order")
    time.sleep(5)

    step_number_list = []  # 横坐标数值
    dev_f1_list = []
    test_f1_list = []
    for res in res_all:
        step_num_str = suffix_find(res['step'], args_global.CHECKPOINT_FOLDER_PREFIX)
        try:
            step_num = int(step_num_str)
        except ValueError:  # 字符串无法int化
            print(f"ValueError: invalid literal for int(): '{step_num_str}', ignore this step")
            continue
        step_number_list.append(step_num)
        dev_f1_list.append(res['triple_dev']['f1'])
        if "triple_test" in res:
            test_f1_list.append(res['triple_test']['f1'])

    # paint
    plt.plot(step_number_list, dev_f1_list, label='dev_f1', color='blue')
    if len(test_f1_list) == len(step_number_list):
        plt.plot(step_number_list, test_f1_list, label='test_f1', color='red')
    plt.legend()            # 添加图例，即标注各曲线分别代表什么。
    plt.title(f"{file_name} f1 change")  # 图的名字
    plt.xlabel('step')     # x轴名字
    plt.ylabel('f1')      # y轴名字
    plt.grid(True)       # 网格线
    pic_name = "curve(f1)_" + file_name[:-5] + ".png"  # json -> png
    plt.savefig(os.path.join(file_path, pic_name))
    print("curve saved")


if __name__ == "__main__":
    print(args_global)
    timepoint_start = time.time()
    res_path_name = score_all()
    print(f"get score END. take time: {(time.time() - timepoint_start) / 60} mins")

    # paint curve of f1
    # res_path_name = "outputs/LSTM compare in LossCRF/240725_LSTM-BiL2H576_LossCRF/score_triple_complete.json"
    # paint_f1_curve(res_path_name)

    # # test. 获取rouge分数
    # get_rouge_test(
    #     pred_txt="支持GPRS的CS3和CS4编码，实现信道编码功能",
    #     label_txt="支持GPRS的CS3和CS4编码",)

