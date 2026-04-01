
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

There are additional parameters at the beginning of the evaluate_1_checkpoint__for_TPLinker function.
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
""" ^^^ STATISTIC_RANGE: Statistical range
    -- all: Compare all triples in all samples, output p, r, f1
    -- segmented_entity: Compare triples with segmented entities in all samples. p, r, f1 results are meaningful.
                    Where p uses the number of triples with segmented entities predicted by the model as the denominator;
    -- triple_num(?,??): Select samples where the number of triples is in the range ? <= triple_num < ??. p, r, f1 results are meaningful.
    -- EPO: Statistic entity pair overlapping, but the sample size for this case is too small, so reference value is limited.
    -- SEO_sample: Select samples containing SingleEntityOverlapping. p, r, f1 results are meaningful.
    -- SEO_triple: Only count triples with SingleEntityOverlapping. Only r is meaningful.
    -- sent_len(?,??): Select samples where sentence length is in the range ? <= sent_len < ??. p, r, f1 results are meaningful.
    -- entity_len(?,??): Only count triples where entity length is within a certain range. p, r, f1 results are meaningful.
    -- segment_num(?,??): Only count triples where entity segment count is within a certain range. Only r is meaningful.
    -- pred_del_seg_ent: Delete triples with segmented entities predicted by ourmodel
    
    以上参数可进行组合，能进一步减小统计范围
"""

# ---------- 预测文件
parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/LSTM compare in LossCE/240725_LSTM-BiL2H576_LossCE")
# parser.add_argument('--OUTPUT_DIR', type=str, default="outputs/nyt/nyt_LSTM-BiL2H576_LossCE")
parser.add_argument('--CHECKPOINT_FOLDER_PREFIX', type=str, default="checkpoint-epoch")  # Prefix of checkpoint folders excluding numeric parts
parser.add_argument('--PREDICT_FILENAME_dev', type=str, default="predict_triples_dev.txt")
parser.add_argument('--PREDICT_FILENAME_test', type=str, default="predict_triples_test.txt")
parser.add_argument('--llm_output_group', type=str, default="0,1,2,3,4,5,6,7,8,9")
# ^^^ Special for integrating multiple outputs from large models. Takes effect when MODEL_NAME in ['t5']

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


args_global = parser.parse_args()

if args_global.OUTPUT_DIR[-1] == '\n':
    args_global.OUTPUT_DIR = args_global.OUTPUT_DIR[:-1]
if args_global.OUTPUT_DIR[-1] == '\r':   # Remove possible carriage return characters
    args_global.OUTPUT_DIR = args_global.OUTPUT_DIR[:-1]
LLM_Output_Group = args_global.llm_output_group.split(",")


def process_after_BertTokenizer_decode(text_decode):  # jyz add 2024-07
    """
    tokenizer = BertTokenizerFast.from_pretrained(
        run_args.model_dir, additional_special_tokens=added_token, do_basic_tokenize=False,
        add_special_tokens=True, do_lower_case=True)
    When decoding with the above tokenizer, there will be spaces between Chinese characters, and the leading "##" will not be removed. Therefore, manual processing is required.
    """
    a2z = "abcdefghijklmnopqrstuvwxyz"
    text = ""
    # Manually remove spaces between tokens, but keep spaces between English words
    for i in range(len(text_decode)):
        if text_decode[i] == " ":
            if text_decode[i - 1] in a2z and text_decode[i + 1] in a2z:
                text += text_decode[i]
        else:
            text += text_decode[i]
    # Remove special characters from the beginning and end
    text = text.replace("[CLS]", "")
    text = text.replace("[SEP]", "")
    text = text.strip("#")
    # Replace [UNK] with a question mark
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
    Pad the left side of the string with 'char' until it reaches max_len
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
    :param no_symbol_l_str_in_sub_str: Whether the substring should not contain `span_l_str`
        If set to False, when input_str="abcdab--yz", span_l_str="ab", span_r_str="yz", 
            sub_str="cdab--". When set to True, sub_str="--"
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
        # Truncate the prefix to ensure the substring does not contain span_l_str
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
    # Check if there is overlap
    # Case 1: span1's end value is greater than span2's start value, and span1's start value is less than span2's end value
    # Case 2: span2's end value is greater than span1's start value, and span2's start value is less than span1's end value
    # If either case is true, there is overlap

    # Extract start and end values of the spans
    x1, y1 = span1
    x2, y2 = span2
    return (x1 < y2 and x2 < y1) or (x2 < y1 and x1 < y2)


def if_label_triple_num_in_range(triple_list: list, range1: tuple, ):
    """
    Function designed to limit statistical range. Check if the number of triples in a sample is within the specified range
    :param triple_list:
    :param range1:  (1,2) slicing rule
    :return:
    """
    if range1[0] <= len(triple_list) < range1[1]:
        return True
    else:
        return False


def if_sent_len_in_range(sent, tokenizer, range1: tuple, ):
    """
    Function designed to limit statistical range.
    Check if the sentence length (number of tokens generated by BertTokenizer) is within the specified range
    :param triple_list:
    :param range1:  (1,2) slicing rule
    :return:
    """
    tokens_id = tokenizer.encode(sent, add_special_tokens=False)
    if range1[0] <= len(tokens_id) < range1[1]:
        return True
    else:
        return False


def if_entity_len_in_range(entity_list, tokenizer, range1: tuple, ):
    """
    Function designed to limit statistical range.
    Check if the entity length (number of tokens generated by BertTokenizer) is within the specified range
    :param range1:  (1,2) slicing rule
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
    Function designed to limit statistical range.
    Check if the number of entity segments is within the specified range
    :param range1:  (1,2) slicing rule
    :return:
    """
    if range1[0] <= segment_num < range1[1]:
        return True
    else:
        return False



def if_sample_has_EPO(triple_list: list, ):
    """
    Function designed to limit statistical range. Check if the sample contains EntityPairOverlapping cases. Too few samples.
    :param triple_list:
    :return:
    """
    triple_ep_list = []  # Store all entity pairs
    triple_epo_list = []  # Store entity pairs with EPO
    for triple in triple_list:
        triple_ep = {triple['subject'], triple['object']}  # entity pair
        if triple_ep not in triple_ep_list:
            triple_ep_list.append(triple_ep)
        else:
            if triple_ep not in triple_epo_list:  # Add to triple_epo_list
                triple_epo_list.append(triple_ep)
    return triple_epo_list


def if_sample_has_SEO(triple_list):
    """
    Function designed to limit statistical range. Check if the sample contains SingleEntityOverlapping cases.
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
              args, tokenizer):  # Get rouge score
    if args.TOKEN_TYPE == 'jieba':
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
    elif args.TOKEN_TYPE == 'tokenizer':
        pred_tokens = my_text_cut(pred_txt, tokenizer)
        label_tokens = my_text_cut(label_txt, tokenizer)
    if len(pred_tokens) == 0:  # Prevent pred_tokens from being empty
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


def get_rouge_test(pred_txt: str, label_txt: str, tokenizer):  # Get rouge score
    pred_tokens = my_text_cut(pred_txt, tokenizer)
    print(f"pred_tokens = {pred_tokens}")
    label_tokens = my_text_cut(label_txt, tokenizer)
    print(f"label_tokens = {label_tokens}")
    if len(pred_tokens) == 0:  # Prevent pred_tokens from being empty
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
        Considering rouge, determine if triple_pred exists in triples_label_remain.
        1. Find the most similar triple_label to triple_pred in triples_label_remain
            The similarity criterion is: relations must be the same, and the smaller rouge score between subject and object is maximized
        2. If the most similar rouge score is greater than the threshold, consider triple_pred exists in triples_label_remain
        3. If triple_pred exists in triples_label_remain, return True and remove the corresponding triple from triples_label_remain; otherwise, return False and return triples_label_remain
    """
    assert len(triple_pred) == 3, f"\n{triple_pred}"

    best_label = {"triple_i": None, "rouge_score": 0}
    for triple_i in range(len(triples_label_remain)):
        triple_label = triples_label_remain[triple_i]
        # relation not same
        if triple_pred['rela'] != triple_label['rela']:
            continue
        # Get rouge score between pred and label subjects
        subj_rouge_score = get_rouge(triple_pred['subj'], triple_label['subj'],
                                     args=args, tokenizer=tokenizer)[0]
        # Get rouge score between pred and label objects
        obj_rouge_score = get_rouge(triple_pred['obj'], triple_label['obj'],
                                    args=args, tokenizer=tokenizer)[0]
        # Rouge score fusion strategy 1: take the smaller of the subject and object WHICH_ROUGE scores
        triple_score = min(subj_rouge_score[args.WHICH_ROUGE]['f'],
                           obj_rouge_score[args.WHICH_ROUGE]['f'])
        # 更新best
        if triple_score > best_label['rouge_score']:
            best_label['rouge_score'] = triple_score
            best_label['triple_i'] = triple_i
            if best_label['rouge_score'] > 0.99:  # == 1.0
                break

    return best_label


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
    # Relevant and predicted correctly,       Total predicted relevant,            Actually relevant
    gold_no = 0  # Actually not this label
    # guess_is_lbl_when_gold_is_lbl = 0  # Predicted as this label in samples that are actually this label
    guess_have_when_gold_no = 0  # Predicted as relevant in samples that are actually not relevant

    for i in range(len(preds)):

        # Get triples for each sample
        triples_pred = preds[i]
        triples_label = labels[i]  # [(s, r, o), ...]

        guess_have += len(triples_pred)
        gold_have += len(triples_label)

    if args.USE_ROUGE is False:
        # Normal judgment of correctness
        for triple_pred in triples_pred:
            assert triples_pred.count(triple_pred) == 1, f"\n{preds[i]}\n{labels[i]}"
            if triple_pred in triples_label:
                correct_have += 1
    else:
        # Add rouge for correctness judgment
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
    # Relevant and predicted correctly,       Total predicted relevant,            Actually relevant
    gold_no = 0  # Actually not this label
    # guess_is_lbl_when_gold_is_lbl = 0  # Predicted as this label in samples that are actually this label
    guess_have_when_gold_no = 0  # Predicted as relevant in samples that are actually not relevant

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
    return p_micro, r_micro, f1_micro


def evaluate_1_checkpoint__for_ourmodel(predict_file, label_file, tokenizer):
    from dataset_loader import relation_modify

    # Evaluate and score data from one checkpoint (train, dev, or test)
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
        }    # Output for complex scenarios

        # -------------------- Process label
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
                triple_info['subj'] = subj_str.split()[-1].lower()  # Take the last word as entity
                triple_info['rela'] = relation_modify(rela_str, mode='nyt')
                triple_info['obj'] = obj_str.split()[-1].lower()
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # Take the last word as entity
                triple_info['rela'] = relation_modify(rela_str, mode='webnlg')
                triple_info['obj'] = obj_str.split()[-1].lower()
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- Prediction processing
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
                triple_info['subj'] = subj_str.split()[-1].lower()  # Take the last word as entity
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str.split()[-1].lower()
            elif 'webnlg' in args_global.DATASET_DIR:
                triple_info['subj'] = subj_str.split()[-1].lower()  # Take the last word as entity
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
        }    # Output for complex scenarios

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
            else:
                triple_info['subj'] = subj_str
                triple_info['rela'] = rela_str
                triple_info['obj'] = obj_str
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- Prediction processing
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

    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())

    assert len(predict_data) == len(label_data)

    # -------------------- Iterate through label_data to get structured sample information
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # Iterate through labels
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # Output for complex scenarios

        # -------------------- Label processing
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

            # 丢弃 offset!=0 的样本
            del predict_data[sample_i]

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
    # Evaluate and score data from one checkpoint (train, dev, or test)
    with open(predict_file, 'r', encoding='UTF-8') as f:
        predict_data = json.loads(f.read())
    with open(label_file, 'r', encoding='UTF-8') as f:
        label_data = json.loads(f.read())
    assert len(predict_data) == len(label_data)

    # -------------------- Iterate through label_data to get structured sample information
    all_samples_triples_label = []
    all_samples_triples_pred = []
    all_samples_triples_pred_label_output = []
    for d_i in range(len(label_data)):  # Iterate through labels
        sample_out = {
            'text': label_data[d_i]['text'],
            'triples_label': label_data[d_i]['relation_list'].copy(),
            'triples_pred': [],
        }    # output for complex scenarios

        # -------------------- Label processing
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

        # -------------------- Label processing
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
        # Remove remaining "##" from both sides and delete spaces in the string
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
        }    # Output for complex scenarios

        # -------------------- Label processing
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

            # String processing to align with flawed predictions
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

        # -------------------- Prediction processing
        triples_pred_in = predict_data[d_i]['triple_list_pred'].copy()
        triples_pred = []
        for triple in triples_pred_in:
            subj_str = triple[0]
            rela_str = triple[1]
            obj_str = triple[2]

            # String processing
            subj_str = subj_str.replace('[UNK]', '?')
            obj_str = obj_str.replace('[UNK]', '?')

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
    # Evaluate and score data from one checkpoint (train, dev, or test)
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

        # -------------------- Label processing
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

        # -------------------- Prediction processing
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
            # ^^^ Remove elements with certain values.
            if len(appear_group_list) * 2 <= len(LLM_Output_Group):
                continue    # Consider incorrect if less than or equal to half

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
    if args_global.STATISTIC_RANGE == "all":  # No need to count complex scenarios
        return

    all_samples_triples_label = []
    all_samples_triples_pred = []
    for sample in samples_label_pred:  # Iterate through labels
        text = sample['text']

        # Complex scenarios filtering
        if "triple_num" in args_global.STATISTIC_RANGE:
            triple_value = eval(span_find(args_global.STATISTIC_RANGE, "triple_num", ")")[0] + ")")
            if if_label_triple_num_in_range(sample['triples_label'], triple_value) is False:
                continue
        # if "sent_len" in args_global.STATISTIC_RANGE:
        #     if if_sent_len_in_range(
        #             label_data[d_i]['text'], tokenizer,
        #             eval(args_global.STATISTIC_RANGE[8:])) is False:
        #         continue
        if 'EPO' in args_global.STATISTIC_RANGE:  # EPO situation, too few samples
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

        # -------------------- Label processing
        triples_label = []
        for triple_str_pos in sample['triples_label']:
            subj_str = triple_str_pos['subject']
            rela_str = triple_str_pos['predicate']
            obj_str = triple_str_pos['object']
            subj_char_span = triple_str_pos['subj_char_span']
            obj_char_span = triple_str_pos['obj_char_span']

            # Complex scenarios filtering
            if "segmented_entity" in args_global.STATISTIC_RANGE:
                # Remove triples without segmented entities
                if len(subj_char_span) == 1 and len(obj_char_span) == 1:
                    continue
            if "segment_num" in args_global.STATISTIC_RANGE:
                segment_num = max([len(subj_char_span), len(obj_char_span)])
                range_value = eval(span_find(args_global.STATISTIC_RANGE, "segment_num", ")")[0] + ")")
                if if_segment_num_in_range(segment_num, range_value) is False:
                    continue
            if "SEO_triple" in args_global.STATISTIC_RANGE:  # Count recall rate of triples with SEO
                if subj_str not in entity_seo_list and obj_str not in entity_seo_list:
                    continue
            if "entity_len" in args_global.STATISTIC_RANGE:
                value = eval(span_find(args_global.STATISTIC_RANGE, "entity_len", ")")[0] + ")")
                if if_entity_len_in_range([subj_str, obj_str], tokenizer, value) is False:
                    continue

            triple_info = {'subj': subj_str, 'rela': rela_str, 'obj': obj_str}
            if triple_info not in triples_label:
                triples_label.append(triple_info)

        # -------------------- Prediction processing
        triples_pred = []
        for triple_str in sample['triples_pred']:
            subj_str = triple_str['subj']
            rela_str = triple_str['rela']
            obj_str = triple_str['obj']

            # Complex scenarios filtering
            if "entity_len" in args_global.STATISTIC_RANGE:
                value = eval(span_find(args_global.STATISTIC_RANGE, "entity_len", ")")[0] + ")")
                if if_entity_len_in_range([subj_str, obj_str], tokenizer, value) is False:
                    continue
            if "segmented_entity" in args_global.STATISTIC_RANGE:
                # Remove triples without segmented entities
                if 'subj_char_span' in triple_str:
                    if len(triple_str['subj_char_span']) == 1 and len(triple_str['obj_char_span']) == 1:
                        continue
                else:
                    if subj_str in text and obj_str in text:
                        continue
            if "pred_del_seg_ent" in args_global.STATISTIC_RANGE:
                # Remove all triples with segmented entities from predictions
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

    print(f"-- Complex scenario: {args_global.STATISTIC_RANGE}")
    print(f"    sample_num={len(all_samples_triples_label)}")
    print(f"    label_triple_num={sum([len(triples) for triples in all_samples_triples_label])}, "
          f"pred_triple_num={sum([len(triples) for triples in all_samples_triples_pred])}")
    p, r, f1 = f1_score_triple(all_samples_triples_pred, all_samples_triples_label,
                               args=args_global, tokenizer=tokenizer)
    res_triple = {'p': round(p, 4), 'r': round(r, 4), 'f1': round(f1, 4)}
    print(f"    {res_triple}")


def score_all():

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

    # Tokenizer setup
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

    # Get checkpoint directories
    checkpoint_dir_list = []
    folders = list(os.walk(args_global.OUTPUT_DIR))[0][1]
    for folder in folders:
        if args_global.CHECKPOINT_FOLDER_PREFIX in folder:
            checkpoint_dir_list.append(os.path.join(args_global.OUTPUT_DIR, folder))
    checkpoint_dir_list.sort(
        key=lambda x: str_left_add(suffix_find(x, args_global.CHECKPOINT_FOLDER_PREFIX), "0", 9))
    print(f"checkpoint_dir_list = {checkpoint_dir_list}\n")
    print(f"Length of checkpoint_dir_list is {len(checkpoint_dir_list)}")
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
            print(f"Triple dev: {res_triple_dev}")
            evaluate_complex_scenarios(samples_label_pred_dev, tokenizer)

        predict_file = os.path.join(checkpoint_dir, args_global.PREDICT_FILENAME_test)
        label_file = os.path.join(args_global.DATASET_DIR, args_global.LABEL_FILENAME_test)
        if os.path.isfile(predict_file) is True:
            res_triple_test, samples_label_pred_test = evaluate_1_checkpoint(
                predict_file=predict_file, label_file=label_file, tokenizer=tokenizer,
            )
            res['triple_test'] = res_triple_test
            print(f"Triple test: {res_triple_test}")
            evaluate_complex_scenarios(samples_label_pred_test, tokenizer)
            # evaluate_complex_scenarios(samples_label_pred_test+samples_label_pred_dev, tokenizer)  # for EPO

        if 'triple_dev' in res:  # and 'triple_test' in res:
            res_all.append(res)
        else:
            print("    This checkpoint does not have complete case output, cannot score")

    res_all.sort(key=lambda x: x['triple_dev']['f1'], reverse=True)

    # Determine the best checkpoint
    best_checkpoint = res_all[0]['step']

    res_all.append({
        # 'aver_top10_f1_dif': f1_dif,
        'best_checkpoint': best_checkpoint,  # For prediction
    })    # Additional information

    # Complex scenarios
    if args_global.STATISTIC_RANGE != "all":
        print("Complex scenario statistics, no save and curve.")
        exit()  # Do not save

    # Save results
    rouge_suffix = "complete"
    if args_global.USE_ROUGE:
        rouge_suffix = f"{args_global.WHICH_ROUGE.replace('-', '')}" \
                       f"({args_global.ROUGE_THRE})"

    file_name = f"score_triple_{rouge_suffix}.json"
    file_name_TestTop = f"score_triple(TestTop)_{rouge_suffix}.json"

    with open(os.path.join(args_global.OUTPUT_DIR, file_name), "w", encoding="utf-8") as fp:
        json.dump(res_all, fp, ensure_ascii=False, indent=4)
    print(f"score_triple.json saved")
    return os.path.join(args_global.OUTPUT_DIR, file_name)


def paint_f1_curve(file_path_name):
    print("Painting F1 curve")

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
    print("Check the order")
    time.sleep(5)

    step_number_list = []  # X-axis values
    dev_f1_list = []
    test_f1_list = []
    for res in res_all:
        step_num_str = suffix_find(res['step'], args_global.CHECKPOINT_FOLDER_PREFIX)
        try:
            step_num = int(step_num_str)
        except ValueError:  # Cannot convert string to int
            print(f"ValueError: invalid literal for int(): '{step_num_str}', ignore this step")
            continue
        step_number_list.append(step_num)
        dev_f1_list.append(res['triple_dev']['f1'])
        if "triple_test" in res:
            test_f1_list.append(res['triple_test']['f1'])

    # Plotting
    plt.plot(step_number_list, dev_f1_list, label='dev_f1', color='blue')
    if len(test_f1_list) == len(step_number_list):
        plt.plot(step_number_list, test_f1_list, label='test_f1', color='red')
    plt.legend()            # Add legend to label each curve
    plt.title(f"{file_name} f1 change")  # Plot title
    plt.xlabel('step')     # X-axis label
    plt.ylabel('f1')      # Y-axis label
    plt.grid(True)       # Show grid lines
    pic_name = "curve(f1)_" + file_name[:-5] + ".png"  # json -> png
    plt.savefig(os.path.join(file_path, pic_name))
    print("Curve saved")


if __name__ == "__main__":
    print(args_global)
    timepoint_start = time.time()
    res_path_name = score_all()
    print(f"Get score END. Time taken: {(time.time() - timepoint_start) / 60} mins")

    # Paint F1 curve
    # res_path_name = "outputs/LSTM compare in LossCRF/240725_LSTM-BiL2H576_LossCRF/score_triple_complete.json"
    # paint_f1_curve(res_path_name)

    # # Test. Get rouge score
    # get_rouge_test(
    #     pred_txt="支持GPRS的CS3和CS4编码，实现信道编码功能",
    #     label_txt="支持GPRS的CS3和CS4编码",)

