import time
from lib.callback.progressbar import ProgressBar
import random
import torch
import json


class Histogram:
    """
    直方图相关类
    """

    #
    def __init__(self, left_lim, right_lim, interval, init_show: str = ""):
        self.statistic_info = []
        self.statistic_info_simple = []  # 直接显示这个即可
        left = left_lim  # 每一柱的左边界
        while left < right_lim:
            right = right_lim if left + interval >= right_lim else left + interval
            col_info = [left, right, 0, 0.]  # 左边界，右边界，个数，占比。!!!!!!!!!!!!!
            # 边界规则：[)，最后一个区间规则：[]
            col_info_simple = [round(left, 2), 0, 0.]  # 左边界，个数， 占比
            self.statistic_info.append(col_info.copy())
            self.statistic_info_simple.append(col_info_simple.copy())
            left = right
        self.left_lim = left_lim
        self.right_lim = right_lim
        self.sample_in_lim_num = 0
        self.over_left_num = 0
        self.over_right_num = 0
        # print("-- a histogram has been initialized: {}".format(init_show))
        # print(self.statistic_info_simple)

    def input_one_data(self, data):  # 直方图统计时添加一个数据
        if data < self.left_lim:
            self.over_left_num += 1
            return
        if data > self.right_lim:
            self.over_right_num += 1
            return
        self.sample_in_lim_num += 1
        for i in range(len(self.statistic_info) - 1, -1, -1):
            if self.statistic_info[i][0] <= data <= self.statistic_info[i][1]:  # [l, r)
                self.statistic_info[i][2] += 1
                break

    def update_ratio(self):  # 直方图显示前更新比率
        # sample_num = 0
        # for col_info in self.statistic_info:
        #     sample_num += col_info[2]

        sample_num = self.sample_in_lim_num
        if sample_num <= 0:  # 防止零除错误
            sample_num = 1

        for i in range(len(self.statistic_info)):
            self.statistic_info[i][3] = float(self.statistic_info[i][2]) / sample_num
            self.statistic_info_simple[i][2] = round(self.statistic_info[i][3], 4)
            self.statistic_info_simple[i][1] = self.statistic_info[i][2]


class Char_Token_SpanConverter(object):
    """
    用于数据集生成准确的 token_char_mapping, 并互化
    version 241006:
        -- 在 get_tok_span 中添加了对 char_span 的校验
    version 240825: 添加 返回mapping的函数
    version 240725 : 考虑了span互化时，输入span为(x,x)的异常情况，print了一些提示信息。
    """

    def __init__(self, tokenizer, add_special_tokens=False, has_return_offsets_mapping=True):
        """
        add_special_tokens: 如果 add_special_tokens=True，会将 [CLS] 考虑在内，token_span 数值整体+1
        has_return_offsets_mapping: bool. tokenizer自身是否包含return_offsets_mapping功能，若不包含，由spanconverter生成。
        """
        self.tokenizer = tokenizer
        self.token_info = None
        self.error_tok_spans = []  # {text, char_span, char_span_str, tok_span_str}
        self.add_special_tokens = add_special_tokens  # 不影响 tokenizer 初始化时设置的 add_special_tokens
        self.has_return_offsets_mapping = has_return_offsets_mapping

    def get_tok_span(self, text: str, char_span):

        # check
        assert char_span[1] > char_span[0] >= 0, f"\n{text}\n{char_span}"

        # get mapping
        self._get_mapping(text)

        # get token span
        if char_span[0] == char_span[1]:
            token_span = self._get_tok_span((char_span[0], char_span[1] + 1))
            token_span = (token_span[0], token_span[0])
            print(f"\n-- Char_Token_SpanConverter.get_tok_span\n"
                  f"    get tok_span={token_span} by char_span={char_span} in \nsent={text}")
        else:  # normal situation
            token_span = self._get_tok_span(char_span)

        # # check
        # self._char_tok_span_check(char_span, token_span)
        return tuple(token_span)

    def get_char_span(self, text: str, token_span):
        # get mapping
        self._get_mapping(text)

        # get char span
        if token_span[0] == token_span[1]:
            char_span_list = self.token_info["tok2char_mapping"][token_span[0]:token_span[1] + 1]
            char_span = (char_span_list[0][0], char_span_list[0][0])
            print(f"\n-- Char_Token_SpanConverter.get_char_span\n"
                  f"    get char_span={char_span} by tok_span={token_span} in \nsent={text}")
        else:  # normal situation
            char_span_list = self.token_info["tok2char_mapping"][token_span[0]:token_span[1]]
            char_span = (char_span_list[0][0], char_span_list[-1][1])

        return char_span

    def get_mapping_tok2char(self, text):
        self._get_mapping(text)
        return self.token_info["tok2char_mapping"]  # 满足切片规则

    def get_mapping_char2tok(self, text):
        self._get_mapping(text)
        return self.token_info["char2tok_mapping"]

    def _get_mapping(self, text):
        """
        实际返回 encode_plus 生成的 token相关信息，其中添加了一些key，主要包括 char2tok_mapping
        """
        if self.token_info is not None and self.token_info["text"] == text:
            return  # 跳过重复操作

        if self.has_return_offsets_mapping is True:
            # Tokenizer 自带生成 offset_mapping(tok2char_mapping) 的功能
            token_info = self.tokenizer.encode_plus(text,
                                                    return_offsets_mapping=True,
                                                    add_special_tokens=self.add_special_tokens)
            token_info["text"] = text  # 添加原文
            token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])

            tok2char_span = token_info["offset_mapping"]
            token_info["tok2char_mapping"] = tok2char_span.copy()
            del token_info["offset_mapping"]

            char_num = None
            for tok_ind in range(len(tok2char_span) - 1, -1, -1):
                if tok2char_span[tok_ind][1] != 0:
                    char_num = tok2char_span[tok_ind][1]
                    break
            char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
            for tok_ind, char_sp in enumerate(tok2char_span):
                for char_ind in range(char_sp[0], char_sp[1]):
                    tok_sp = char2tok_span[char_ind]
                    # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                    if tok_sp[0] == -1:
                        tok_sp[0] = tok_ind
                    tok_sp[1] = tok_ind + 1
            token_info["char2tok_mapping"] = char2tok_span.copy()

        else:  # self.has_return_offsets_mapping is False
            token_info = self.tokenizer.encode_plus(text,
                                                    add_special_tokens=self.add_special_tokens)
            token_info["text"] = text  # 添加原文
            token_info["tokens"] = self.tokenizer.convert_ids_to_tokens(token_info["input_ids"])

            # ---------------------------------------- get char2tok_mapping
            tokens = token_info["tokens"].copy()
            char2tok_mapping = [(-1, -1)] * len(text)
            tokens_i = [0, 0]  # 起始：下标为0的token的下标为0的字符
            if tokens[0] == self.tokenizer.cls_token:
                tokens_i = [1, 0]  # 起始：下标为1的token的下标为0的字符
            # 遍历字符
            for c_i, c in enumerate(text):
                c_belong_unk = 0
                c_tokens = self.tokenizer.tokenize(c)
                if len(c_tokens) == 0:  # c 是一个空白字符
                    pass
                else:
                    ct = c_tokens[0]
                    # 查找字符在哪个token中
                    while ct not in tokens[tokens_i[0]]:
                        if tokens[tokens_i[0]] == '[UNK]' and ct not in tokens[tokens_i[0] + 1]:
                            c_belong_unk = 1
                            break
                        tokens_i[0] += 1
                        tokens_i[1] = 0
                        assert tokens_i[0] < len(tokens), f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
                    if ct == '[UNK]':
                        c_belong_unk = 1

                    if c_belong_unk == 0:
                        # 查找字符在token中哪个位置
                        ct_pos = tokens[tokens_i[0]].find(ct, tokens_i[1])
                        assert ct_pos >= tokens_i[1], f"\n{text}\n{tokens}\n{tokens_i}\n{c_i}\n{ct}"
                        # 添加到char2tok_mapping
                        char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
                        # 更新tokens_i
                        tokens_i[1] = ct_pos + len(ct)
                        if tokens_i[1] >= len(tokens[tokens_i[0]]):
                            tokens_i[0] += 1
                            tokens_i[1] = 0
                    else:
                        char2tok_mapping[c_i] = (tokens_i[0], tokens_i[0] + 1)
            token_info["char2tok_mapping"] = char2tok_mapping.copy()

            # ---------------------------------------- get tok2char_mapping
            tok2char_mapping = [(-1, -1)] * len(tokens)
            for c_i in range(len(text)):
                if char2tok_mapping[c_i][0] == -1 or char2tok_mapping[c_i][0] == char2tok_mapping[c_i][1]:
                    continue
                token_i = char2tok_mapping[c_i][0]
                if tok2char_mapping[token_i] == (-1, -1):
                    tok2char_mapping[token_i] = (c_i, c_i + 1)
                else:
                    assert c_i + 1 > tok2char_mapping[token_i][1]
                    tok2char_mapping[token_i] = (tok2char_mapping[token_i][0], c_i + 1)
            token_info["tok2char_mapping"] = tok2char_mapping.copy()

        self.token_info = token_info
        # return token_info

    def _get_tok_span(self, char_span):
        """
        得到 tok_span
        """
        # char2tok_span: 列表，每个元素表示每个句中字符对应的token下标。
        #   每个元素一般取值为[a,a+1]，
        #   如果连续多个元素位于一个token中，则会出现`[a,a+1],[a,a+1],...`，
        #   如果是例如空格等字符，不会出现在token中，则取值[-1,-1]

        tok_span_list = self.token_info["char2tok_mapping"][char_span[0]:char_span[1]]
        tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
        return tok_span

    def _char_tok_span_check(self, char_span, tok_span):
        """
        校验 tok_span 是否能抽取出与 char_span 一样的文本
        token_info: 必须包含 text, input_ids
        tokenizer: 必须是生成 token_info 的 tokenizer
        char_span: 长度为2的列表或元组，暂时不考虑分段情况
        tok_span: 长度为2的列表或元组，暂时不考虑分段情况
        """
        sub_text_from_char0 = self.token_info['text'][char_span[0]:char_span[1]]
        sub_text_from_char = self.tokenizer.decode(self.tokenizer.encode(sub_text_from_char0, add_special_tokens=False))

        sub_text_from_token = self.tokenizer.decode(self.token_info['input_ids'][tok_span[0]:tok_span[1]])

        if sub_text_from_char == sub_text_from_token:
            return True
        else:
            error_tok_span = {
                'text': self.token_info['text'],
                'char_span': char_span,
                'char_span_str': sub_text_from_char,
                'tok_span_str': sub_text_from_token
            }
            if error_tok_span not in self.error_tok_spans:
                self.error_tok_spans.append(error_tok_span)
                print(f"char_span string: [{sub_text_from_char0}][{sub_text_from_char}], but tok_span string: [{sub_text_from_token}]")
            return False


# RELATION_SET = ["功能", "手段采用", "前提是", "造成", "影响",
#                 "分类", "组成部分", "属性有", "含有",
#                 "定义", "别名", "实例为", "特点", ]
# 标签列表
NER_TAG_LIST = ["X", "O",
                "Head-I", "Head-IO", "Head-B1", "Head-B2", "Head-B3", "Head-B4", "Head-B5",
                "Tail-I", "Tail-IO", "Tail-B1", "Tail-B2", "Tail-B3", "Tail-B4", "Tail-B5",
                ]

SPECIAL_TOKENS = {
    'diy_str': {'rela': '[rela]',
                'subj': '[subj]',
                'obj': '[obj]',
                'entity_omit': '[ent_omit]', },
}

ADDITIONAL_SPECIAL_TOKENS = ['[rela]', '[subj]', '[obj]', '[ent_omit]', '“', '”']


# def which_list_find_small_first(list1, list2):
#     # 在哪个列表中先找到较小数字
#     list1.append(min(list1) - 100)  # 添加哨兵
#     list2.append(min(list2) - 100)  # 添加哨兵
#     i_max = len(list1) if len(list1) < len(list2) else len(list2)
#     i = 0
#     while i < i_max:
#         if list1[i] < list2[i]:
#             return 1
#         elif list2[i] < list1[i]:
#             return 2
#         i += 1
#     return 0  # 两序列完全相同


def triples_sort__priority_judge(triple1_info, triple2_info):
    """
    该函数在判断优先级，用于排序。
    排序：一个样本（一句话）中包含多个实体关系三元组，将它们按在文中出现的顺序排序。
        第一优先级：主体位置。主体位置完全相同时判断第二优先级
        第二优先级：客体位置
    :param triple1_info: [ [列表元素是主体各字符位置下标], [列表元素是客体各字符位置下标] ]
    :param triple2_info: 同上
    :return:
    """

    t1_subj, t1_obj = triple1_info
    t1__char_pos_idx = t1_subj + [-1] + t1_obj + [-1]
    """
        通过将主客体位置连接，将两个优先级连接起来。
        但不能直接将位置连接，需要加 -1 作为哨兵，这是为了防止bug（例如，
            triple1_info = [[0,1,2,3], [10,11]]，triple2_info = [[0], [10,11,12,13]]，
            可看出triple2_info优先级更高，但如果直接将位置连接，会导致判断triple1_info优先级更高）
    """
    t2_subj, t2_obj = triple2_info
    t2__char_pos_idx = t2_subj + [-1] + t2_obj + [-1]

    i_max = len(t1__char_pos_idx) if len(t1__char_pos_idx) < len(t2__char_pos_idx) else len(t2__char_pos_idx)
    i = 0
    while i < i_max:
        if t1__char_pos_idx[i] < t2__char_pos_idx[i]:
            return 1  # triple1 先出现位置较前的字符，triple1 优先级高
        elif t2__char_pos_idx[i] < t1__char_pos_idx[i]:
            return 2
        i += 1
    return 0  # 两序列完全相同


def triples_insert_sort_once(triples, triple_new):
    """
    该函数执行插入排序的一轮，即将一个样本插入合适的位置。
    排序：一个样本（一句话）中包含多个实体关系三元组，将它们按在文中出现的顺序排序。
    :param triples:   一个按优先级由高到低排列的三元组列表
        example: triples = [
                [('LTE中传输块的信道编码方案', '实例为', 'Turbo编码'), [(12, 26)], [(27, 34)]],
                [('LTE', '含有', '传输块'), [(12, 15)], [(16, 19)]],
                [('传输块', '属性有', '信道编码方案'), [(16, 19)], [(20, 26)]],
                ...
            ]
    :param triple_new:   一个新三元组，等待被插入
    :return: triples
    """

    triple_new_subj = [ele for the_tuple in triple_new[1] for ele in range(the_tuple[0], the_tuple[1])]
    """
        example: input 'triples[i][1] = [(0, 5), (10, 16)]', 
                output 'triple_i_subj = [0,1,2,3,4,10,11,12,13,14,15]'  及主体每个字符在句子中的下标
    """
    triple_new_obj = [ele for the_tuple in triple_new[2] for ele in range(the_tuple[0], the_tuple[1])]
    triple_new_info = [triple_new_subj, triple_new_obj]

    for i in range(len(triples)):
        triple_i_subj = [ele for the_tuple in triples[i][1] for ele in range(the_tuple[0], the_tuple[1])]
        triple_i_obj = [ele for the_tuple in triples[i][2] for ele in range(the_tuple[0], the_tuple[1])]
        # ^^^ 将每个实体分段的跨度（切片）表示展开成列举出所有实体下标
        triple_i_info = [triple_i_subj, triple_i_obj]
        res = triples_sort__priority_judge(triple_i_info, triple_new_info)
        if res == 1:  # 新三元组优先级低，继续比较
            continue
        elif res == 2:  # 新三元组优先级高，就插入此位置
            triples.insert(i, triple_new.copy())  # 无返回值，不能return
            return triples
        else:  # == 0:
            assert 0, f"\n三元组中有两个位置完全一样的\n  {triples}\n  {triple_new}\n  {i}"
    triples.insert(len(triples), triple_new.copy())
    return triples


def sample_triples_sort(triples):
    """
    一个样本（一句话）中包含多个实体关系三元组，将它们按在文中出现的顺序排序
    :param triples:
        example: triples = [
                [('LTE中传输块的信道编码方案', '实例为', 'Turbo编码'), [(12, 26)], [(27, 34)]],
                [('LTE', '含有', '传输块'), [(12, 15)], [(16, 19)]],
                [('传输块', '属性有', '信道编码方案'), [(16, 19)], [(20, 26)]],
                ...
            ]
    :return:
    """

    if len(triples) <= 1:
        return triples

    triples_sort = []
    for triple in triples:
        assert len(triple) >= 3 and (len(triple) - 3) % 2 == 0, f"\n{triple}"  # 长度大于3是句子中含有多对相同的三元组的情况
        i = 1
        while i < len(triple):
            # if i > 1:
            #     print(triple)
            trip = [triple[0], triple[i], triple[i + 1]].copy()
            triples_sort = triples_insert_sort_once(triples_sort, trip)  # 执行一次插入排序
            i += 2
    return triples_sort


def sample__get_1_relation(sent, relation, triples):
    """
    样本按关系细分，并按实体在句子中出现的顺序排序。
    example of 1 triple: [('LTE下行资源映射方式', '分类', '集中式'), [(0, 5), (10, 16)], [(17, 20)]]
    :return:
    """

    triples_1rela = []
    triples_1rela__str_only = []
    for triple in triples:
        assert len(triple) == 3, f"\n{sent}\n{triple}"
        subj_rela_obj = triple[0]
        if subj_rela_obj[1] == relation and subj_rela_obj not in triples_1rela__str_only:
            # 如果句子中含有多对相同的三元组，只取最先出现的一个。
            triples_1rela.append(triple)
            triples_1rela__str_only.append(subj_rela_obj)

    # triples_1rela = []
    # for triple in triples:
    #     subj_rela_obj__tuple = triple[0]
    #     if subj_rela_obj__tuple[1] == relation:
    #         if len(triple) == 3:
    #             triples_1rela = triples_insert_sort_once(triples_1rela, triple)
    #         else:  # 句中含有多处相同三元组的情况
    #             # print("\n句中含有多处相同三元组的情况")
    #             # print(triple)
    #             assert len(triple) > 3 and (len(triple)-3)%2==0, f"\n{sent}\n{triple}"
    #             triple_same = []
    #             i = 1
    #             while i < len(triple):
    #                 triple_same = triples_insert_sort_once(triple_same, [triple[0], triple[i], triple[i + 1]])
    #                 i += 2
    #             # print(f"调整顺序与格式后变为：{triple_same}")
    #             # print(f"只取第一个：{triple_same[0]}")
    #             # time.sleep(5)
    #             triples_1rela = triples_insert_sort_once(triples_1rela, triple_same[0])

    triples_1rela = sample_triples_sort(triples_1rela)  # 排序

    return [sent, relation, triples_1rela].copy()


def sent_token_cut(spanconverter, sent, max_token_len):  # 将sent按max_token_len截短
    spanconverter.get_tok_span(sent, (0, 1))
    sent_token_info = spanconverter.token_info
    if len(sent_token_info['input_ids']) > max_token_len:
        max_char_len = 0
        token_i = max_token_len
        while max_char_len <= 0:
            max_char_len = sent_token_info['tok2char_mapping'][token_i][1]
            token_i -= 1
        sent = sent[:max_char_len]
    return sent


def sent_add_prompt(prompt_mode, sent, rela=None, rela_mode=None, triple_last=None,
                    max_entity_char_len=None):
    """

    :param sent:
    :param rela:
    :param triple_last:  [(subj, rela, obj), [subj_pos], [obj_pos]] or []
                triple_last is [] when extracting the first triple
    :param max_entity_char_len:
    :return:
    """
    assert prompt_mode in ['all_text', 'entity_emb', 'all_emb']
    sent_with_prompt = sent

    if prompt_mode in ['all_emb']:
        return sent_with_prompt

    # 添加关系提示
    if prompt_mode in ['all_text', 'entity_emb']:
        if rela_mode in ['sep_normal']:
            rela_prompt = f" {SPECIAL_TOKENS['diy_str']['rela']} {rela}"
        elif rela_mode in ['symbol']:     # 自定义的special token
            rela_prompt = f" {SPECIAL_TOKENS['diy_str']['rela']} [{rela}]"
        else:
            assert 0, rela_mode
        sent_with_prompt += rela_prompt

    # 添加实体提示
    if prompt_mode in ['all_text']:
        ent_side = int(max_entity_char_len / 2)  # 提示实体一侧的最长长度
        subj_last = ''
        obj_last = ''
        if len(triple_last) > 0:
            subj_last = triple_last[0][0]
            obj_last = triple_last[0][2]
        if subj_last and obj_last:
            if len(subj_last) > max_entity_char_len:  # 实体太长时，省略中间内容
                subj_last = subj_last[:ent_side] + f" {SPECIAL_TOKENS['diy_str']['entity_omit']} " + subj_last[-ent_side:]
            if len(obj_last) > max_entity_char_len:
                obj_last = obj_last[:ent_side] + f" {SPECIAL_TOKENS['diy_str']['entity_omit']} " + obj_last[-ent_side:]
            sent_with_prompt += f" {SPECIAL_TOKENS['diy_str']['subj']} " + subj_last + \
                                f" {SPECIAL_TOKENS['diy_str']['obj']} " + obj_last  # 添加上一个主客体提示

    return sent_with_prompt


def sample_format_trans__sent1rela_to_sent1label(sample_1_rela):
    """
    输入的是一个样本句子，及其一个关系，及该关系对应的三元组。
    将输入转化为一个句子对应一个三元组标签的格式，并在句子中添加提示
        input =     ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。', '组成部分', [
                    [('GSM无线控制信道', '组成部分', 'BCCH'), [(1, 4), (6, 12)], [(14, 18)]],
                    [('GSM无线控制信道', '组成部分', 'CCCH'), [(1, 4), (6, 12)], [(19, 23)]],
                    [('GSM无线控制信道', '组成部分', 'DCCH'), [(1, 4), (6, 12)], [(24, 28)]]]]
                    or
                    ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。', '属性有', []]

    :return:
    """

    sent, rela, triples = sample_1_rela.copy()
    sample_sent1label_list = []
    triple_i = 0
    while triple_i <= len(triples):

        # get this triple as label
        if triple_i == len(triples):
            triple = []
        else:
            triple = triples[triple_i]
        # get last triple
        if triple_i > 0:
            triple_last = triples[triple_i - 1]
        else:
            triple_last = []

        # # add prompt to sent
        # if prompt_mode in ["1"]:
        #     sent_prompt = sent_add_prompt(sent, rela, triple_last, max_entity_len)
        #     sample_new = [sent_prompt, triple]
        # elif prompt_mode in ["2"]:
        #     pass
        sample_new = {
            'sent_origin_id': -1,
            'sent_origin': sent,
            'rela': rela,
            'triple_last': triple_last.copy(),
            'triple_label': triple.copy(),  # [(subj, rela, obj), [subj_pos], [obj_pos]] or []
        }

        # add to list
        sample_sent1label_list.append(sample_new.copy())
        triple_i += 1

    assert len(sample_sent1label_list) == len(triples) + 1, \
        f"\n  {sample_1_rela}\n  {sample_sent1label_list}"
    return sample_sent1label_list


def ner_tag_strategy(tag_list, triple_with_tok_pos, strategy='1'):
    """
    :param triple_with_tok_pos: [(subj, rela, obj), [subj_tok_pos], [obj_tok_pos]]
                                    ([] already been filtered out)
    :param tag_list: the tag_list already has 'O' and 'X' tags, ready to set entity tags
    :param strategy:
        1: 当实体被分为多段时，例如被分为 [s1, s2, s3], [o1, o2, o3]
            当s1、s2在文中按顺序出现，且中间没有其他s?时，  仅s1首字母标B1，  s2首字母标I，   s1、s2之间标O
            当s1、s2在文中按顺序出现，但中间有其他s?时，  s1首字母标B1，  s2首字母标B2，
            当s1、s2在文中出现顺序颠倒，              s1首字母标B1，  s2首字母标B2。
            当s1、s2在文中按顺序出现，但中间有其他o?时，  依然是 仅s1首字母标B1，s2首字母标I，s1、s2之间o?的部分标Tail-B/I，其他部分标O
    :return:
    """

    def is_other_span_between_two_span(tup_lst, index):
        # 检查输入的有效性
        if not isinstance(tup_lst, list) or len(tup_lst) < 2 or index < 1 or index >= len(tup_lst):
            raise ValueError("Invalid input.")

        # tup_lst = [(1,2), (9,11)]
        lst = [tup[0] for tup in tup_lst]
        # lst = [1, 9]

        # 获取两个下标对应的元素值
        value1 = lst[index - 1]
        value2 = lst[index]
        assert value1 < value2, f"\n{lst}, {index}"

        # 遍历列表，查找是否存在位于两者之间的值（不包括min_value和max_value）
        for ii in range(len(lst)):
            if ii != index - 1 and ii != index and value1 < lst[ii] < value2:
                return 1

        return 0

    two_ent_pos__subj_first = triple_with_tok_pos[1] + triple_with_tok_pos[2]
    two_ent_pos__obj_first = triple_with_tok_pos[2] + triple_with_tok_pos[1]

    if strategy == '1':
        ent_pos = triple_with_tok_pos[1]  # subj
        tag_prefix = "Head"
        begin_id = 1
        for pos_i in range(len(ent_pos)):
            if pos_i == 0:
                tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                    tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
            else:
                if ent_pos[pos_i][0] < ent_pos[pos_i - 1][0]:  # s1、s2在文中出现顺序颠倒，
                    begin_id += 1
                    tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                    for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
                elif is_other_span_between_two_span(ent_pos, pos_i):  # s1、s2在文中按顺序出现，但中间有其他s?
                    begin_id += 1
                    tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                    for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
                else:  # 其他情况
                    for i in range(ent_pos[pos_i][0], ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
        ent_pos = triple_with_tok_pos[2]  # obj
        tag_prefix = "Tail"
        begin_id = 1
        for pos_i in range(len(ent_pos)):
            if pos_i == 0:
                tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                    tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
            else:
                if ent_pos[pos_i][0] < ent_pos[pos_i - 1][0]:  # s1、s2在文中出现顺序颠倒，
                    begin_id += 1
                    tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                    for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
                elif is_other_span_between_two_span(ent_pos, pos_i):  # s1、s2在文中按顺序出现，但中间有其他s?
                    begin_id += 1
                    tag_list[ent_pos[pos_i][0]] = NER_TAG_LIST.index(f'{tag_prefix}-B{begin_id}')
                    for i in range(ent_pos[pos_i][0] + 1, ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
                else:  # 其他情况
                    for i in range(ent_pos[pos_i][0], ent_pos[pos_i][1]):
                        tag_list[i] = NER_TAG_LIST.index(f'{tag_prefix}-I')
    return tag_list


def ner_tag_encode(
        span_converter: Char_Token_SpanConverter,
        sent_with_prompt: str, sent_tokens_len: int, prompt_tok_pos: int, triple_with_char_pos: list,
        strategy: str = '1',
):
    """

    :param span_converter:
    :param sent_with_prompt:
    :param sent_tokens_len:
    :param triple_with_char_pos: [(subj, rela, obj), [subj_char_pos], [obj_char_pos]] or []
    :param prompt_tok_pos: the token position of '[rela]'
    :param strategy:
    :return:
    """
    tag_ids = [NER_TAG_LIST.index('O')] * sent_tokens_len
    for i in range(prompt_tok_pos, sent_tokens_len):
        tag_ids[i] = NER_TAG_LIST.index('X')

    if len(triple_with_char_pos) == 0 or \
            triple_with_char_pos[0][0] == "" or triple_with_char_pos[0][2] == "":
        return tag_ids

    # 将主客体的char_span转变为token_span
    subj_char_pos = triple_with_char_pos[1]
    subj_tok_pos = [span_converter.get_tok_span(sent_with_prompt, span_slice)
                    for span_slice in subj_char_pos]
    obj_char_pos = triple_with_char_pos[2]
    obj_tok_pos = [span_converter.get_tok_span(sent_with_prompt, span_slice)
                   for span_slice in obj_char_pos]
    triple_with_token_pos = [triple_with_char_pos[0], subj_tok_pos, obj_tok_pos]
    # 标注tag
    tag_ids = ner_tag_strategy(tag_ids, triple_with_token_pos, strategy=strategy)

    return tag_ids


def ner_tag_decode(
        span_converter: Char_Token_SpanConverter, sent: str, rela: str, tag_list, strategy='1'):
    """
    单句tag解码
    注意 输入 token_list、tag_list 之间是否有 [CLS]、[SEP] 需要对应（即要么都有，要么都没有，程序中不包含判别）
    token_list 是否有 [CLS]、[SEP]，需要在 span_converter 中设置。
    :param span_converter: 其中包含 tokenizer
    :param strategy: reverse process in 'ner_tag_strategy()'
    :return:
    """

    def continuous_slices(lst):
        """
        将列表中一一列举的连续的数字变成切片的左右数值，若出现不连续的位置则生成下一个切片
        例如输入[1,2,3,4,10,11,12,6,7,9] ，则输出[(1,5), (10,13), (6,8), (9,10)]
        """
        if not lst:
            return []
        result = []
        start = lst[0]
        end = start + 1
        for num in lst[1:]:
            if num == end:
                end += 1
            else:
                result.append((start, end))
                start = num
                end = start + 1
        result.append((start, end))  # 不要忘记添加最后一个连续序列
        return result

    def char_pos_2_str(sent, char_pos):
        ent = ""
        for span in char_pos:
            ent += sent[span[0]:span[1]]
        return ent

    def tag2entity_strategy1(tag_prefix):
        ent_pos_list = [[] for _ in range(5)]
        slice_id = 1
        # print("")
        for c_i, tag_num in enumerate(tag_list):
            if f'{tag_prefix}-B' in NER_TAG_LIST[tag_num]:
                slice_id_last = slice_id
                slice_id = int(NER_TAG_LIST[tag_num][6])
                if len(ent_pos_list[slice_id - 1]) == 0:  # B? tag 只读取第一个
                    ent_pos_list[slice_id - 1].append(c_i)
                else:
                    slice_id = slice_id_last
            elif NER_TAG_LIST[tag_num] == f'{tag_prefix}-I':
                ent_pos_list[slice_id - 1].append(c_i)
            # print(NER_TAG_LIST[tag_num])
            # print(ent_pos_list)
        pos_each_char = [ele for sublist in ent_pos_list for ele in sublist]  # 二重嵌套列表铺平
        ent_tok_spans = continuous_slices(pos_each_char)
        ent_char_spans = [span_converter.get_char_span(sent, span) for span in ent_tok_spans]
        for span_i in range(len(ent_char_spans) - 1, -1, -1):  # 删除无内容的span
            if ent_char_spans[span_i][0] == ent_char_spans[span_i][1]:
                del ent_char_spans[span_i]
        ent_str = char_pos_2_str(sent, ent_char_spans)
        return ent_tok_spans, ent_char_spans, ent_str

    # get relation
    # rela = ''
    # rela_pos = sent.find(SPECIAL_TOKENS['diy_str']['rela']) + len(SPECIAL_TOKENS['diy_str']['rela'])
    # assert rela_pos > 0
    # for offset in range(1, 10):
    #     if sent[rela_pos:rela_pos+offset] in RELATION_SET:
    #         rela = sent[rela_pos:rela_pos+offset]
    #         break
    # assert rela in RELATION_SET, f"\n{sent}\n{rela}"

    # subj = ''
    # subj_pos = []
    # obj = ''
    # obj_pos = []
    triple_info = {}

    if strategy == '1':
        # tag_prefix = "Head"  # 查找 subj
        # # 以下是subj、obj操作相同部分
        # ent_pos_list = [[] for _ in range(5)]
        # slice_id = 1
        # for c_i, tag_num in enumerate(tag_list):
        #     if f'{tag_prefix}-B' in NER_TAG_LIST[tag_num]:
        #         slice_id = int(NER_TAG_LIST[tag_num][6])
        #         # ent_list[slice_id - 1] += token_list[c_i]
        #         ent_pos_list[slice_id - 1].append(c_i)
        #     elif NER_TAG_LIST[tag_num] == f'{tag_prefix}-I':
        #         # ent_list[slice_id - 1] += token_list[c_i]
        #         ent_pos_list[slice_id - 1].append(c_i)
        # pos_each_char = [ele for sublist in ent_pos_list for ele in sublist]
        # ent_tok_spans = continuous_slices(pos_each_char)
        # ent_char_spans = [span_converter.get_char_span(sent, span) for span in ent_tok_spans]
        # for span_i in range(len(ent_char_spans)-1, -1, -1):
        #     if ent_char_spans[span_i][0] == ent_char_spans[span_i][1]:  # 切片无内容
        #         del ent_char_spans[span_i]
        # ent_str = char_pos_2_str(sent, ent_char_spans)
        # # 以上是subj、obj操作相同部分
        # subj_pos = continuous_slices(pos_each_char)
        # triple_info['subj_tok_span'] = subj_pos
        # triple_info['subj_char_span'] = [span_converter.get_char_span(sent, span) for span in subj_pos]
        # subj = char_pos_2_str(sent, triple_info['subj_char_span'])
        triple_info['subj_tok_span'], triple_info['subj_char_span'], subj = \
            tag2entity_strategy1(tag_prefix="Head")

        # tag_prefix = "Tail"  # 查找 obj
        # # 以下是subj、obj操作相同部分
        # ent_pos_list = [[] for _ in range(5)]
        # slice_id = 1
        # for c_i, tag_num in enumerate(tag_list):
        #     if f'{tag_prefix}-B' in NER_TAG_LIST[tag_num]:
        #         slice_id = int(NER_TAG_LIST[tag_num][6])
        #         # ent_list[slice_id - 1] += token_list[c_i]
        #         ent_pos_list[slice_id - 1].append(c_i)
        #     elif NER_TAG_LIST[tag_num] == f'{tag_prefix}-I':
        #         # ent_list[slice_id - 1] += token_list[c_i]
        #         ent_pos_list[slice_id - 1].append(c_i)
        # pos_each_char = [ele for sublist in ent_pos_list for ele in sublist]
        # # 以上是subj、obj操作相同部分
        # obj_pos = continuous_slices(pos_each_char)
        # triple_info['obj_tok_span'] = obj_pos
        # triple_info['obj_char_span'] = [span_converter.get_char_span(sent, span) for span in obj_pos]
        # obj = char_pos_2_str(sent, triple_info['obj_char_span'])
        triple_info['obj_tok_span'], triple_info['obj_char_span'], obj = \
            tag2entity_strategy1(tag_prefix="Tail")

        triple_info['triple_str'] = (subj, rela, obj)

    return triple_info


def process_batch(batch):
    """
    作为 Dataloader 的 collate_fn
    :param batch: 一个包含batch_size个字典的列表，
                每个字典都有 input_ids, label_ids, attention_mask, segment_ids, 等成员，
                    具体结构详见 Dataset.samples_1_label__to_bert_input_2406() 的输出.
    :return:
    """

    def list_padding(lst: list, target_len: int, padding_value=0):
        assert len(lst) <= target_len, f"\nbatch={batch}\nlst={lst}\nlen(lst)={len(lst)}\ntarget_len={target_len}"
        return lst.copy() + [padding_value] * (target_len - len(lst))

    # 长度填充
    max_seq_len = max(len(sample['input_ids']) for sample in batch)
    # above: 此处默认每个样本内各成员间长度相等，取了input_ids来获取长度。
    for i in range(len(batch)):
        batch[i]['input_ids'] = list_padding(batch[i]['input_ids'], max_seq_len, )
        batch[i]['label_ids'] = list_padding(batch[i]['label_ids'], max_seq_len, )
        batch[i]['bert_att_mask'] = list_padding(batch[i]['bert_att_mask'], max_seq_len, )
        batch[i]['tag_mask'] = list_padding(batch[i]['tag_mask'], max_seq_len, )
        batch[i]['segment_ids'] = list_padding(batch[i]['segment_ids'], max_seq_len, )
        # batch[i]['label_ids_last'] = list_padding(batch[i]['label_ids_last'], max_seq_len, )

    # 打包，转化为torch.tenser
    input_ids = torch.tensor([sample['input_ids'] for sample in batch], dtype=torch.long)
    bert_att_mask = torch.tensor([sample['bert_att_mask'] for sample in batch], dtype=torch.long)
    segment_ids = torch.tensor([sample['segment_ids'] for sample in batch], dtype=torch.long)
    label_ids = torch.tensor([sample['label_ids'] for sample in batch], dtype=torch.long)
    tag_mask = torch.tensor([sample['tag_mask'] for sample in batch], dtype=torch.long)
    # label_ids_last = torch.tensor([sample['label_ids_last'] for sample in batch], dtype=torch.long)
    # above: 用于triple_last的实体需要用它的tag_list嵌入来表示的情况

    input_len = torch.tensor([sample['input_len'] for sample in batch], dtype=torch.long)
    original_sent_id = torch.tensor([sample['original_sent_id'] for sample in batch], dtype=torch.long)
    # above: 指示该句的原句子是哪一句，用于验证时，将验证集属于同一个样本的所有三元组汇总
    # rela_id = torch.tensor([sample['rela_id'] for sample in batch], dtype=torch.long)
    # above: 用于需要rela_id嵌入的情况

    return {
        'input_ids': input_ids,
        'label_ids': label_ids,
        'bert_att_mask': bert_att_mask,
        'segment_ids': segment_ids,
        'tag_mask': tag_mask,
        # 'label_ids_last': label_ids_last,

        'input_len': input_len,
        'original_sent_id': original_sent_id,
        # 'rela_id': rela_id,
    }


def relation_modify(rela_origin, mode='nyt'):
    A2Z = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    a2z = 'abcdefghijklmnopqrstuvwxyz'

    if mode == 'nyt':
        rela_origin = rela_origin.replace('/', ' ').strip()
        rela = rela_origin.split(' ')[-1].replace('_', ' ').strip()
        return rela

    elif mode == 'webnlg':
        rela = rela_origin.replace('_', ' ')
        for i in range(len(rela) - 2, 0, -1):  # 首字母大写的单词前添加空格
            if rela[i] in A2Z and rela[i - 1] in a2z and rela[i + 1] in a2z:
                rela = rela[:i] + " " + rela[i:]
        rela = rela.strip()
        return rela


class Dataset:
    def __init__(self, file_data, file_rela, tokenizer, args):
        self.args = args
        self.file_data = file_data

        # relation types
        with open(file_rela, "r", encoding="utf-8") as fi:
            relation_dict = eval(fi.read())
        self.relation_list = list(relation_dict.keys())
        if 'nyt' in file_data:  # 针对nyt数据集的关系词的调整
            for i in range(len(self.relation_list)):
                self.relation_list[i] = relation_modify(self.relation_list[i], mode='nyt')
        elif 'webnlg' in file_data:
            for i in range(len(self.relation_list)):
                self.relation_list[i] = relation_modify(self.relation_list[i], mode='webnlg')

        with open(file_data, "r", encoding="utf-8") as fi:
            samples = eval(fi.read())
        # self.samples = samples
        # 为了与历史程序兼容，需要在此转换格式
        self.samples = []
        for sample_form_in in samples:
            triple_list = []
            for triple_form_in in sample_form_in['relation_list']:

                rela = triple_form_in['predicate']
                if 'nyt' in file_data:  # 针对nyt数据集的关系词的调整
                    rela = relation_modify(rela, mode='nyt')
                elif 'webnlg' in file_data:  # 针对webnlg数据集的关系词的调整
                    rela = relation_modify(rela, mode='webnlg')

                if type(triple_form_in['subj_char_span'][0]) == list or type(triple_form_in['subj_char_span'][0]) == tuple:
                    # for CMIM2023-KG-task1-Re
                    subj_char_span = [tuple(span) for span in triple_form_in['subj_char_span']]
                    obj_char_span = [tuple(span) for span in triple_form_in['obj_char_span']]
                elif type(triple_form_in['subj_char_span'][0]) == int and len(triple_form_in['subj_char_span']) == 2:
                    # 兼容传统实体不分段的数据集
                    subj_char_span = [tuple(triple_form_in['subj_char_span'])]
                    obj_char_span = [tuple(triple_form_in['obj_char_span'])]

                triple_form_out = [(triple_form_in['subject'],
                                    rela,
                                    triple_form_in['object']),
                                   subj_char_span.copy(),
                                   obj_char_span.copy()]
                triple_list.append(triple_form_out)
            sample_form_out = [sample_form_in['id'], sample_form_in['text'], triple_list.copy()]
            self.samples.append(sample_form_out)
        """ samples[i] 
            格式改为了下面的！！！
            ['train-142', '在Atoll中， Timeslot表中定义每个小区的每个时隙的上行负载因子、下行总功率、other CCH信道功率', [
                [('Atoll', '含有', ' Timeslot表'), [(1, 6)], [(8, 18)]], 
                [('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的上行负载因子'), [(1, 18)], [(21, 37)]], 
                [('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的下行总功率'), [(1, 18)], [(21, 31), (38, 43)]], 
                [('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的other CCH信道功率'), [(1, 18)], [(21, 31), (44, 57)]], 
                [('小区', '含有', '时隙'), [(23, 25)], [(28, 30)]], 
                ]], 
        """

        self.tokenizer = tokenizer
        self.char_token_spanconverter = Char_Token_SpanConverter(tokenizer, add_special_tokens=True)

    def samples_1_label__to_bert_input_2406(self, datas):
        """

        :param datas: a list
                datas[?] = {
                    'sent_origin_id': ?,
                    'sent_origin': sent,
                    'rela': rela,
                    'triple_last': triple_last.copy(),
                    'triple_label': triple.copy(),   # [(subj, rela, obj), [subj_char_pos], [obj_char_pos]] or []
                }
                triple_label is only useful while training.
        :return:
        """

        # datas_bert = []
        datas_bert = [{} for _ in range(len(datas))]
        # pbar = ProgressBar(n_total=len(datas), desc=f'trans to bert input')
        for datas_i in range(len(datas)):
            # pbar(datas_i)
            if datas_i + 1 % 100000 == 0:
                print(f"    [trans to bert input format] - {datas_i + 1}/{len(datas)}")
            # original_sent_id, sent_with_prompt, triple = datas[datas_i].copy()
            sent_with_prompt = sent_add_prompt(
                prompt_mode=self.args.triple_last_fuse_mode, sent=datas[datas_i]['sent_origin'],
                rela=datas[datas_i]['rela'], rela_mode=self.args.rela_prompt_mode,
                triple_last=datas[datas_i]['triple_last'],
                max_entity_char_len=self.args.max_entity_char_len)

            triple_with_char_pos = datas[datas_i]['triple_label'].copy()

            sep_rela = SPECIAL_TOKENS['diy_str']['rela']
            assert sep_rela in ADDITIONAL_SPECIAL_TOKENS, f"\n{ADDITIONAL_SPECIAL_TOKENS}"
            prompt_char_pos = sent_with_prompt.find(sep_rela)
            prompt_token_pos = self.char_token_spanconverter.get_tok_span(
                sent_with_prompt, (prompt_char_pos, prompt_char_pos + len(sep_rela)))  # for setting tag_mask
            sent_token_info = self.char_token_spanconverter.token_info
            """ sent_token_info example:
            {
            'input_ids': [101, 3844, 7030, 2845, 1440, 3221, 1762, 8945, 8860, 8199, 928, 6887, 677, 837, 6843, 4638, 21128, 1216, 5543, 102], 
            'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'text': '测量报告是在SACCH信道上传送的 [rela] 功能', 
            'tokens': ['[CLS]', '测', '量', '报', '告', '是', '在', 'sa', '##cc', '##h', '信', '道', '上', '传', '送', '的', '[rela]', '功', '能', '[SEP]'], 
            'tok2char_mapping': [(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 21), (21, 22), (22, 23), (0, 0)], 
            'char2tok_mapping': [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [7, 8], [8, 9], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [16, 17], [16, 17], [16, 17], [17, 18], [18, 19]]}
            """
            if sent_token_info.get('segment_ids') is None:
                sent_token_info['segment_ids'] = sent_token_info['token_type_ids'].copy()
                del sent_token_info['token_type_ids']
            if sent_token_info.get('bert_att_mask') is None:
                sent_token_info['bert_att_mask'] = sent_token_info['attention_mask'].copy()
                del sent_token_info['attention_mask']

            # 添加 label str
            if len(triple_with_char_pos) > 0:
                sent_token_info['label'] = triple_with_char_pos[0]
            else:
                sent_token_info['label'] = ()

            # 添加tag
            # if "通常用阻塞干扰来衡量接收机抗邻道干扰的能力。" in sent_with_prompt:
            #     print("")
            #     print(f"-- sent_with_prompt = {sent_with_prompt}")
            #     print(f"-- triple_with_char_pos = {triple_with_char_pos}")
            sent_token_info['label_ids'] = ner_tag_encode(
                span_converter=self.char_token_spanconverter, sent_with_prompt=sent_with_prompt,
                sent_tokens_len=len(sent_token_info['input_ids']), prompt_tok_pos=prompt_token_pos[0],
                triple_with_char_pos=triple_with_char_pos, strategy='1',
            )
            # if "通常用阻塞干扰来衡量接收机抗邻道干扰的能力。" in sent_with_prompt:
            #     print(f"-- tag_list = {sent_token_info['label_ids']}")

            # 校验一下
            # triple_decode = ner_tag_decode(span_converter, sent_with_prompt, tag_ids, strategy='1')
            # if triple_decode['subj_tok_span'] != triple_token_pos[1] or triple_decode['obj_tok_span'] != triple_token_pos[2]:
            #     print(f"tag解码可能存在问题")
            #     print(f"tokens = {sent_token_info['tokens']}")
            #     print(f"tag_ids = {tag_ids}")
            #     print(f"实际 token span = {triple_token_pos}")
            #     print(f"解码 token span = {triple_decode}")
            #     time.sleep(2)

            # 添加 tag_mask
            tag_mask = [0] * len(sent_token_info['input_ids'])
            for i in range(prompt_token_pos[0]):
                tag_mask[i] = 1
            sent_token_info['tag_mask'] = tag_mask.copy()

            # 获取上一个 triple 的 tag list （用于其他的信息融合方式）
            # if "通常用阻塞干扰来衡量接收机抗邻道干扰的能力。" in sent_with_prompt:
            #     print(f"-- triple_with_char_pos last = {datas[datas_i]['triple_last']}")
            sent_token_info['label_ids_last'] = ner_tag_encode(
                span_converter=self.char_token_spanconverter, sent_with_prompt=sent_with_prompt,
                sent_tokens_len=len(sent_token_info['input_ids']), prompt_tok_pos=prompt_token_pos[0],
                triple_with_char_pos=datas[datas_i]['triple_last'], strategy='1',
            )
            # if "通常用阻塞干扰来衡量接收机抗邻道干扰的能力。" in sent_with_prompt:
            #     print(f"-- tag_list last = {sent_token_info['label_ids_last']}")

            # 添加token序列长度
            sent_token_info['input_len'] = len(sent_token_info['input_ids'])

            # 添加原始句子id
            sent_token_info['original_sent_id'] = datas[datas_i]['sent_origin_id']

            # 添加关系id
            sent_token_info['rela_id'] = self.relation_list.index(datas[datas_i]['rela'])  # 用于rela需要embedding的情况
            assert sent_token_info['rela_id'] > -1, f"\n{datas[datas_i]}\n{self.relation_list}"

            sent_token_info = dict(sent_token_info)
            # ^^^ 若不进行dict()转化，sent_token_info是
            #     <class 'transformers.tokenization_utils_base.BatchEncoding'>类型，无法json序列化

            # 为了减少空间，必须删掉无用的成员
            del sent_token_info['text']
            del sent_token_info['tok2char_mapping']
            del sent_token_info['char2tok_mapping']
            del sent_token_info['label']
            del sent_token_info['label_ids_last']
            del sent_token_info['rela_id']

            # datas_bert.append(sent_token_info)
            datas_bert[datas_i] = sent_token_info.copy()
            #

        """ samples_bert[?] = 
        {
        'original_sent_id': 0
        'label': ('TD-LTE上行受限的信道', '实例为', '业务信道/PUSCH'), 
        'tokens': ['[CLS]', 'T', 'D', '-', 'L', 'T', 'E', '上', '行', '受', '限', '的', '信', '道', '是', '业', '务', '信', '道', '/', 'P', 'U', 'S', 'C', 'H', '##关', '实', '例', '为', '[SEP]'], 
        'input_ids': [101, 162, 146, 118, 154, 162, 147, 677, 6121, 1358, 7361, 4638, 928, 6887, 3221, 689, 1218, 928, 6887, 120, 158, 163, 161, 145, 150, 14125, 2141, 891, 711, 102], 
        'label_ids': [1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 11, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1], 
        'tag_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        'input_len': 30, 
        'bert_att_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ...
        }
        """
        return datas_bert

    def format__bert(self, args):
        """
        将数据的格式转换为bert输入格式
        """
        print("-- Dataset.format__bert(): processing dataset to the format of `1 sent to 1 label`")
        samples_1_label = []
        for sent_id in range(len(self.samples[:])):
            # id_, sent, triples_dict = samples[sample_i].copy()
            # triples_list = list(triples_dict.values())
            id_, sent, triples_list = self.samples[sent_id].copy()
            """ triples_list
            [[('TD-LTE上行受限的信道', '实例为', '业务信道/PUSCH'), [(0, 13)], [(14, 24)]], 
            [('TD-LTE', '属性有', '上行受限的信道'), [(0, 6)], [(6, 13)]]]
            """

            # 将句子长度截短，超出句子长度处的triple也删除
            sent = sent_token_cut(self.char_token_spanconverter, sent, args.max_origin_sent_token_len)
            triples_list_cut = []
            for triple in triples_list:
                if max([span_slice[1] for span_slice in triple[1]] + [span_slice[1] for span_slice in triple[2]]) <= len(sent):
                    triples_list_cut.append(triple.copy())

            for relation in self.relation_list:  # 按关系细分
                # ------------------------------ 调整为 1 sent, 1 rela, labels
                sample_1_rela = sample__get_1_relation(sent, relation, triples_list_cut)
                """ sample_1_rela
                ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。', '组成部分', [
                [('GSM无线控制信道', '组成部分', 'BCCH'), [(1, 4), (6, 12)], [(14, 18)]], 
                [('GSM无线控制信道', '组成部分', 'CCCH'), [(1, 4), (6, 12)], [(19, 23)]], 
                [('GSM无线控制信道', '组成部分', 'DCCH'), [(1, 4), (6, 12)], [(24, 28)]]]]
                or
                ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。', '属性有', []]
                """
                # ------------------------------ 调整为 1 sent, 1 rela, 1 label
                sample_1_rela_1_label__list = sample_format_trans__sent1rela_to_sent1label(sample_1_rela)
                """ sample_1_rela_1_label__list
                [
                ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。【关系分隔】组成部分', 
                    [('GSM无线控制信道', '组成部分', 'BCCH'), [(1, 4), (6, 12)], [(14, 18)]] ],
                ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。【关系分隔】组成部分【主体分隔1】GSM无线控制信道【客体分隔1】BCCH', 
                    [('GSM无线控制信道', '组成部分', 'CCCH'), [(1, 4), (6, 12)], [(19, 23)]] ],
                ...
                ['在GSM中，无线控制信道包括BCCH、CCCH、DCCH。【关系分隔】功能', 
                    []],
                ]
                """

                # samples_1_label += sample_1_rela_1_label__list.copy()
                for sample_1label in sample_1_rela_1_label__list:
                    sample_1label['sent_origin_id'] = sent_id  # 添加 原句子id
                    samples_1_label.append(sample_1label.copy())
                    # samples_1_label.append([sent_id] + sample_1label.copy())

            # if i == 10:
            #     print(XXXXXX)

        del triples_list_cut
        del sample_1_rela
        del sample_1_rela_1_label__list

        print(f"-- the number of `sent with prompt` = `label number` = {len(samples_1_label)}")
        """ samples_1_label = [
        ...
        [6, '参数ASSOC用于控制启动“指配到其他小区允许”功能。[rela]组成部分', []], 
        [6, '参数ASSOC用于控制启动“指配到其他小区允许”功能。[rela]属性有', []], 
        [6, '参数ASSOC用于控制启动“指配到其他小区允许”功能。[rela]含有', []], 
        ... ]
        """

        # # 数据集数据量太大的情况下，采用将处理好的数据存放到磁盘，然后直接读取的策略。
        # for i in range(8):
        #     samples_bert = self.samples_1_label__to_bert_input_2406(
        #         samples_1_label[int(len(samples_1_label)*i/8):int(len(samples_1_label)*(i+1)/8)])
        #     with open(f"train_data__bert-base-cased__part{i+1}.json", "w", encoding="utf-8") as fp:
        #         json.dump(samples_bert, fp, ensure_ascii=False)
        #     print(f"saved train_data_bert_part{i+1}.json")
        #     del samples_bert
        # print(xxxxx)
        samples_bert = self.samples_1_label__to_bert_input_2406(samples_1_label)
        """ samples_bert[?] =    (OUTPUT)
        {
        'original_sent_id': 0
        'label': ('TD-LTE上行受限的信道', '实例为', '业务信道/PUSCH'), 
        'tokens': ['[CLS]', 'T', 'D', '-', 'L', 'T', 'E', '上', '行', '受', '限', '的', '信', '道', '是', '业', '务', '信', '道', '/', 'P', 'U', 'S', 'C', 'H', '##关', '实', '例', '为', '[SEP]'], 
        'input_ids': [101, 162, 146, 118, 154, 162, 147, 677, 6121, 1358, 7361, 4638, 928, 6887, 3221, 689, 1218, 928, 6887, 120, 158, 163, 161, 145, 150, 14125, 2141, 891, 711, 102], 
        'label_ids': [1, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 11, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 1, 1, 1], 
        'tag_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 
        'input_len': 30, 
        'bert_att_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
        'segment_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
        """
        for sample in samples_bert:
            # if len(sample['label']) > 0:
            if sample['label_ids'].count(0) + sample['label_ids'].count(1) < len(sample['label_ids']) and \
                    SPECIAL_TOKENS['diy_str']['subj'] in sample['tokens']:
                print(f"\n-- samples_bert example:\n{sample}")
                time.sleep(5)
                break

        return samples_bert

    def format__no_sentid(self):
        """
        去掉句子id，删去字符相同的三元组.
        相比 ”to_triples_stronly_format“ 保留位置标注
        """
        samples_out = []
        for sample_i in range(len(self.samples)):
            id_, sent, triples = self.samples[sample_i].copy()
            triple_str_list = []
            triple_str_pos_list = []
            for triple_str_pos in triples:
                if triple_str_pos[0] not in triple_str_list:  # 文本部分完全相同的三元组只取一个
                    triple_str_list.append(triple_str_pos[0])
                    triple_str_pos_list.append(triple_str_pos.copy())
                else:
                    # print(f"same triple: {triple_str_pos}")
                    pass
            samples_out.append([sent, triple_str_pos_list].copy())
        return samples_out

    def format__no_sentid_pos(self):
        """
        transform to this format:
            samples_out[i] =
                ['在Atoll中， Timeslot表中定义每个小区的每个时隙的上行负载因子、下行总功率、other CCH信道功率', [
                    ('Atoll', '含有', ' Timeslot表'),
                    ('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的上行负载因子'),
                    ('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的下行总功率'),
                    ('Atoll中， Timeslot表', '含有', '每个小区的每个时隙的other CCH信道功率'),
                    ('小区', '含有', '时隙'),
                ]],
        """
        samples_out = []
        for sample_i in range(len(self.samples)):
            id_, sent, triple_str_pos_list = self.samples[sample_i].copy()
            triple_str_list = []
            for triple_str_pos in triple_str_pos_list:
                if triple_str_pos[0] not in triple_str_list:  # 文本部分完全相同的三元组只取一个
                    triple_str_list.append(triple_str_pos[0])
                else:
                    # print(f"same triple: {triple_str_pos}")
                    pass
            samples_out.append([sent, triple_str_list].copy())
        return samples_out

    def format__chatglm3(self, sent_lim=500):
        ques_length = Histogram(0, 1000, 100)
        ans_length = Histogram(0, 2000, 100)

        rela_all = ""
        for rela in self.relation_list:
            rela_all += f"{rela}、"
        rela_all = rela_all[:-1]
        system_info = f"找出句中关系为“{rela_all}”的实体对，若没有则回答无。"

        samples_out = []
        for sample_i in range(len(self.samples)):
            id_, sent, triple_str_pos_list = self.samples[sample_i].copy()
            if 0 < sent_lim < len(sent):
                sent = sent[:sent_lim]
            triple_str_list = []
            last_subj_rela = ([], "")  # subj_pos, rela
            answer = "" if len(triple_str_pos_list) > 0 else "无"
            for triple_str_pos in triple_str_pos_list:
                triple_str = triple_str_pos[0]
                triple_pos = triple_str_pos[1] + triple_str_pos[2]
                triple_pos_r_max = max(triple_pos, key=lambda x: x[1])[1]
                if triple_str not in triple_str_list and triple_pos_r_max < sent_lim:
                    # 文本部分完全相同的三元组只取一个。  在长度限制内
                    triple_str_list.append(triple_str_pos[0])
                    subj, rela, obj = triple_str
                    if triple_str_pos[1] == last_subj_rela[0] and rela == last_subj_rela[1]:
                        answer += '<客体>' + obj
                    else:
                        answer += '<主体>' + subj + '<关系>' + rela + '<客体>' + obj
                        last_subj_rela = (triple_str_pos[1], rela)

            ques_length.input_one_data(len(system_info) + len(sent) + 2)
            ans_length.input_one_data(len(answer) + 1)
            # sample_info = {"conversations": [
            #     {
            #         "role": "system",
            #         "content": system_info
            #     },
            #     {
            #         "role": "user",
            #         "content": sent
            #     },
            #     {
            #         "role": "assistant",
            #         "content": answer
            #     },
            # ]}
            sample_info = {"conversations": [
                {
                    "role": "user",
                    "content": system_info + "<句子>" + sent
                },
                {
                    "role": "assistant",
                    "content": answer
                },
            ]}
            samples_out.append(sample_info)
        ques_length.update_ratio()
        ans_length.update_ratio()
        print(f"问题长度\n{ques_length.statistic_info_simple}  超出：{ques_length.over_right_num}")
        print(f"回答长度\n{ans_length.statistic_info_simple}  超出：{ans_length.over_right_num}")

        return samples_out

    def analysis_relation(self):
        # 每一种关系的占比
        triple_num = 0
        telation_dict = {}
        for rela in self.relation_list:
            telation_dict[rela] = 0
        for id_, sent, triples in self.samples:
            for triple_str_pos in triples:
                triple_str = triple_str_pos[0]
                rela = triple_str[1]
                telation_dict[rela] += 1
                triple_num += 1
        for rela in list(telation_dict.keys()):
            telation_dict[rela] = round(telation_dict[rela] / triple_num, 4)
        print(f"关系占比（...{self.file_data[-20:]}）\n{telation_dict}")

    def analysis_sent_length(self):
        sent_length = Histogram(0, 500, 50)
        for id_, sent, triples in self.samples:
            sent_length.input_one_data(len(sent))
        sent_length.update_ratio()
        print(f"句子长度（...{self.file_data[-20:]}）\n{sent_length.statistic_info_simple}  超出：{sent_length.over_right_num}")

    def analysis_triple_num(self):
        triple_num = 0
        triple_num_discontinues = 0
        for id_, sent, triples in self.samples:
            triple_num += len(triples)
            for triple in triples:
                assert type(triple[1]) == list
                if len(triple[1]) == 1 and len(triple[2]) == 1:
                    pass
                else:
                    triple_num_discontinues += 1
        print(f"三元组个数：{triple_num}")
        print(f"实体不连续的三元组个数：{triple_num_discontinues}")

    def analysis_triple_pos(self):
        # 统计三元组的最靠前的位置位于句子的哪一区间？
        triple_pos = Histogram(0, 500, 50)
        for id_, sent, triples in self.samples:
            for triple_str_pos in triples:
                pos = triple_str_pos[1] + triple_str_pos[2]
                pos_l = 5000
                for (span_l, _) in pos:
                    if span_l < pos_l:
                        pos_l = span_l
                triple_pos.input_one_data(pos_l)
        triple_pos.update_ratio()
        print(f"三元组位置（...{self.file_data[-20:]}）\n{triple_pos.statistic_info_simple}  超出：{triple_pos.over_right_num}")

    def analysis_slice_dist(self):
        # 统计三元组各片段之间的最长距离
        slice_dist = Histogram(0, 100, 10)
        for id_, sent, triples in self.samples:
            for triple_str_pos in triples:
                pos = triple_str_pos[1] + triple_str_pos[2]
                pos_r_min = 5000
                pos_l_max = -1
                for (span_l, span_r) in pos:
                    if span_r < pos_r_min:
                        pos_r_min = span_r
                    if span_l > pos_l_max:
                        pos_l_max = span_l
                dist = pos_l_max - pos_r_min + 1
                assert dist > 0
                slice_dist.input_one_data(dist)
        slice_dist.update_ratio()
        print(f"切片距离（...{self.file_data[-20:]}）\n{slice_dist.statistic_info_simple}  超出：{slice_dist.over_right_num}")

    def analysis_triple_num_each_sample(self):
        # 含有不同的三元组数据量的样本的占比
        # triple_num = Histogram(0,40,5)
        triple_num = Histogram(0, 6, 1)
        for id_, sent, triples in self.samples:
            triple_num.input_one_data(len(triples))
            # if len(triples) == 0:
            #     print(f"-- {sent}")
        triple_num.update_ratio()
        print(f"三元组个数（...{self.file_data[-20:]}）\n{triple_num.statistic_info_simple}  超出：{triple_num.over_right_num}")


if __name__ == "__main__":
    # dataset_divide_train_dev()
    print("END")
