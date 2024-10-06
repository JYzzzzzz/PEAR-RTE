
"""
将 CMIM23-NOM1-RA 数据集转化为 llm 训练验证的格式
"""

import json, random, os


RElATION_LIST = ["功能", "手段采用", "前提是", "造成", "影响",
                 "分类", "组成部分", "属性有", "含有",
                 "定义", "别名", "实例为", "特点"]
RELATION_CH2EN = {
    "功能": "Function", "手段采用": "Approach", "前提是": "Premise", "造成": "Cause", "影响": "Influence",
    "分类": "Category", "组成部分": "Component", "属性有": "Attribute", "含有": "Contain",
    "定义": "Definition", "别名": "Alias", "实例为": "Instance", "特点": "Characteristic",
}

class Histogram:
    """
    直方图相关类

    1、初始化
    2、使用 input_one_data 一个一个添加数据
    3、使用 get_statistic_result 输出统计数据

    version:
        -- 240908: 添加 get_statistic_result 方法
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
        self.update_ratio()

        if simple:
            output = [["(-inf, l_lim)", float(self.smaller_num) / self.sample_in_lim_num]] + \
                     self.statistic_info_simple + \
                     [["(r_lim, inf)", float(self.larger_num) / self.sample_in_lim_num]]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    if type(output[i][j]) not in [str, int]:  # float
                        output[i][j] = round(output[i][j], 2)
            return output
        else:
            output = [["(-inf, l_lim)", self.smaller_num, float(self.smaller_num) / self.sample_in_lim_num]] + \
                     self.statistic_info + \
                     [["(r_lim, inf)", self.larger_num, float(self.larger_num) / self.sample_in_lim_num]]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    if type(output[i][j]) not in [str, int]:  # float
                        output[i][j] = round(output[i][j], 4)
            return output


def length_statistic(samples):
    ques_length = Histogram(0, 1000, 100)
    ans_length = Histogram(0, 1000, 100)

    for sample in samples:
        if 1:  # sample['ans'].count('b) ') == 13:
            ques_length.input_one_data(len(sample['input']))
            ans_length.input_one_data(len(sample['output']))

    ques_length.update_ratio()
    ans_length.update_ratio()
    print(f"-- example: {samples[0]}")
    print(f"-- ques_len: {ques_length.statistic_info_simple}  超出：{ques_length.larger_num}")
    print(f"-- ans_len : {ans_length.statistic_info_simple}  超出：{ans_length.larger_num}")


def random_list_combine(list_in, group_num):
    """
    随机平均将列表中的元素分成几组，有几组会多出一个元素；各组内元素的顺序也是打乱的
    version: 240920
    :param list_in:
    :param group_num:
    :return:
    """
    lst_in = list_in.copy()
    random.shuffle(lst_in)
    ele_num_each_group = len(lst_in) // group_num    # 每组中的元素数量，向下取整。
    group_num__ele_num_add1 = len(lst_in) % group_num  # 多一个元素的组数。

    lst_out = []
    idx = 0
    for _ in range(group_num - group_num__ele_num_add1):
        group = lst_in[idx:idx+ele_num_each_group]
        lst_out.append(group.copy())
        idx += ele_num_each_group
    for _ in range(group_num__ele_num_add1):
        group = lst_in[idx:idx+ele_num_each_group+1]
        lst_out.append(group.copy())
        idx += ele_num_each_group+1
    assert idx == len(lst_in)
    return lst_out


def relation_language_adapt(rela_chinese, lang):
    if lang == 'en':
        return RELATION_CH2EN[rela_chinese]
    else:    # 'ch'
        return rela_chinese


def convert__rela_order__sample_expand(file_name, loop_num=1, lang='ch'):
    """
    提问时，可能只提问其中一部分关系，且关系顺序随机；
    回答时，第一优先级是关系，顺序即提问时出现的顺序；

    进行样本扩充，方便训练，以及进行更精确的预测：
        训练集 1个原样本将扩充为：[[13]*2, [6,7]*2 [4, 4, 5]*2 [3,3,3,4]*2, [2, 2, 2, 2, 2, 3]*1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]*1]

    :return:
    """

    expand_strategy = [[13], [13], [13],
                       [6, 7], [6, 7],
                       [4, 4, 5], [4, 4, 5],
                       [3, 3, 3, 4],
                       [2, 2, 2, 2, 2, 3],
                       [1]*13,
                       ]
    # print(expand_strategy)

    instructions = {
        'part1': {
            'ch': "你是一个中文网络运维领域的关系三元组抽取专家。",
            'en': "You are an expert in extracting relational triples from Chinese network operation and maintenance corpus. ",
        },
        'part_task_desc': {
            'ch': "我会给你一个逻辑关系列表与一个网络运维领域的句子，你要找出句中所有具有列表中关系的主体客体对，",
            'en': "I will provide you with a list of logical relations and a sentence in the field of network operation and maintenance. Your task is to identify all subject-object pairs within the sentence that match the relations in the list. ",
        },
        'part_ans_form': {
            'ch': "按关系顺序以 \"b) 关系: s) 主体 o) 客体 s) 主体 o) 客体 e). \" 的格式回答，没有则回答 \"b) 关系: 无 e). \" 。",
            'en': "Respond in the format \"b) Relation: s) Subject o) Object s) Subject o) Object e). \" following the order of relations in the list. If no pair is found, reply with \"b) Relation: none e). \". ",
        },
        'part_expectation': {
            'ch': "主体客体都是句中的文本，要找全、找准。",
            'en': "Ensure that both subjects and objects are accurately identified and comprehensive from the sentence. ",
        },
        'part_task_start1': {
            'ch': "下面完成实际任务，关系列表为:",
            'en': "Now, let's proceed with the actual task. The relation list is: ",
        },
        'part_task_start2': {
            'ch': "，句子为:",
            'en': ", and the sentence is: ",
        },
        'part_ans_none': {
            'ch': " 无",
            'en': " none",
        },
    }

    with open(file_name, 'r', encoding='UTF-8') as f:
        samples = json.loads(f.read())
        assert type(samples) == list

    samples_out = []
    # ---------------------------------------- samples traversal
    for _ in range(loop_num):
        for d_i in range(len(samples)):
            sent = samples[d_i]['text']
            triple_str_pos_list = samples[d_i]['relation_list']
            # id_, sent, triple_str_pos_list = samples[d_i].copy()

            # ---------------------------------------- sample expanded by different relation combination
            relation_select = []
            for expand_type in expand_strategy:
                relation_select += random_list_combine(RElATION_LIST, len(expand_type))
            # for group_num in [1, 3, 6, len(RELATION_SET)]:   # 将 13 类关系分别分为 1次问答，2次问答，... 13次问答
            #     num = 0
            #     num_max = (float(len(RELATION_SET))/group_num)**0.5  # if for_train is True else 1
            #     while num < num_max:
            #         num += 1
            #         groups = random_list_combine(RELATION_SET, group_num)
            #         relation_select += groups

            # ---------------------------------------- relation_select traversal
            for relation_group_i, relation_set_sub in enumerate(relation_select):

                # ---------------------------------------- 确定要提问的关系列表
                rela_set_str = ""
                for rela in relation_set_sub:
                    rela_set_str += f"{relation_language_adapt(rela, lang)}, "
                rela_set_str = rela_set_str[:-2]   # example: "定义, 特点"

                # ---------------------------------------- 设计指示
                system_info = f"{instructions['part1'][lang]}" \
                              f"{instructions['part_task_desc'][lang]}" \
                              f"{instructions['part_ans_form'][lang]}" \
                              f"{instructions['part_expectation'][lang]}" \
                              f"{instructions['part_task_start1'][lang]}[{rela_set_str}]" \
                              f"{instructions['part_task_start2'][lang]}"

                # ---------------------------------------- design answer
                answer_train = ""
                answer_all_triple = ""
                for rela_need in relation_set_sub:
                    answer_1rela_train = ""    # 用于存放在句子长度限制范围内的（rela_need对应的）主客体对
                    answer_1rela_all_triple = ""   # 存放所有（rela_need对应的）主客体对，不受长度限制
                    triple_str_list = []    # 用于防止重复的三元组加入answer
                    # last_subj_rela = ([], "")  # subj_pos, rela

                    # triples traversal
                    for triple_str_pos in triple_str_pos_list:
                        subj = triple_str_pos['subject']
                        rela = triple_str_pos['predicate']
                        obj = triple_str_pos['object']
                        triple_str = (subj, rela, obj)
                        # triple_pos = triple_str_pos[1] + triple_str_pos[2]
                        # triple_pos_r_max = max(triple_pos, key=lambda x: x[1])[1]

                        # 判断要不要将某三元组加到chatglm3的答案标签中
                        if rela == rela_need and triple_str not in triple_str_list:
                            # 文本部分完全相同的三元组只取一个。  三元组在文中的位置在长度限制内
                            triple_str_list.append(triple_str)
                            answer_1rela_all_triple += ' s) ' + subj + ' o) ' + obj
                            # if triple_pos_r_max <= len(sent_cut):
                            #     answer_1rela_train += '<主体>' + subj + '<客体>' + obj

                    # if answer_1rela_train == "":
                    #     answer_1rela_train = "无"
                    # answer_1rela_train = f"<开始>{rela_need}:{answer_1rela_train}<结束>。"
                    # answer_train += answer_1rela_train
                    if answer_1rela_all_triple == "":
                        answer_1rela_all_triple = instructions['part_ans_none'][lang]
                    answer_1rela_all_triple = f"b) {relation_language_adapt(rela_need, lang)}:{answer_1rela_all_triple} e). "
                    answer_all_triple += answer_1rela_all_triple

                # if len(answer_train) > answer_lim:
                #     answer_train = answer_train[:answer_lim]

                # ---------------------------------------- 封装
                # ques_length.input_one_data(len(system_info) + len(sent) + 2)
                # ans_length.input_one_data(len(answer_train) + 1)
                sample_info = {
                    "instruction": "",
                    "input": system_info + sent,
                    "output": answer_all_triple,
                }
                # if relation_group_i == 0:
                #     sample_info['ans_all_triple'] = answer_all_triple
                samples_out.append(sample_info)
            # ^^^ for relation_group_i, relation_set_sub in enumerate(relation_select):
        # ^^^ for d_i in range(len(samples)):
    #

    return samples_out


def convert__rela_order__sample_expand__oneshot(file_name, loop_num=1, lang='ch'):
    """
    在 convert__rela_order__sample_expand 生成的样本的基础上，在提问中添加一个示例

    t5 tokenizer 的token长度统计 粗略结果 为：
        ques_len = [[300, 0.94], [400, 0.05], [500, 0.01], [600, 0.01], [700, 0.0], [800, 0.0], [900, 0.0], [1000, 0.0], [1100, 0.0], [1200, 0.0]]
        ans_len = [[300, 0.61], [400, 0.17], [500, 0.06], [600, 0.0], [700, 0.06], [800, 0.0], [900, 0.06], [1000, 0.0], [1100, 0.06], [1200, 0.0]]

    :return:
    """

    instructions = {
        'part_real_task': {
            'ch': "下面完成实际任务",
            'en': "Now, let\'s proceed with the actual task",
        },
        'part_rela_list': {
            'ch': "关系列表为:",
            'en': "relation list is: ",   # The --> the
        },
        'part_sent': {
            'ch': "句子为:",
            'en': "and the sentence is: ",
        },
        'part_ans_none2': {
            'ch': "无 e)",
            'en': "none e)",
        },
    }

    # -------------------- 读入示例库
    example_file = "dataset/CMIM23-NOM1-RA_llm_form/rela_order__sample_expand/train.json"
    if lang == 'en':
        example_file = "dataset/CMIM23-NOM1-RA_llm_form/english__rela_order__sample_expand/train.json"
    with open(example_file, 'r', encoding='UTF-8') as f:
        examples = json.loads(f.read())

    samples = convert__rela_order__sample_expand(file_name, loop_num, lang)

    # -------------------- 添加示例
    for d_i in range(len(samples)):
        sample_sent_pos = samples[d_i]['input'].find(instructions['part_sent'][lang]) + len(instructions['part_sent'][lang])
        sample_sent = samples[d_i]['input'][sample_sent_pos:]
        sample_task_pos = samples[d_i]['input'].find(instructions['part_real_task'][lang])

        example_ok = False
        while example_ok is False:
            # -------------------- 选择合适的示例
            example = random.sample(examples, 1)[0]
            example_sent_pos = example['input'].find(instructions['part_sent'][lang]) + len(instructions['part_sent'][lang])
            example_sent = example['input'][example_sent_pos:]
            example_rela_num = example['output'].count("b) ")

            if len(example_sent) > 50 or example_rela_num > 4 or example_sent == sample_sent:
                continue
            if example_rela_num == 1 and random.random() > 4.0/13:
                # 只抽取一个关系的示例，一定概率丢弃
                continue
            if example['output'].count("b) ") == example['output'].count(instructions['part_ans_none2'][lang]) and \
                    random.random() > 0.25:
                # 示例没有抽出任何三元组，一定概率丢弃
                continue

            example_rela_pos = example['input'].find(instructions['part_rela_list'][lang])
            example_str = example['input'][example_rela_pos:]
            if lang == 'ch':
                example_str = "例如，" + example_str + f"，则答案为:{example['output']}。"
            elif lang == 'en':
                example_str = "For example, the " + example_str + f", the answer would be: {example['output']}. "

            samples[d_i]['input'] = samples[d_i]['input'][:sample_task_pos] + example_str + \
                                   samples[d_i]['input'][sample_task_pos:]
            example_ok = True

    return samples


def convert__triple_order__sample_expand__oneshot(file_name, loop_num=1, lang='ch'):

    instructions = {
        'part1': {
            'ch': "你是一个中文网络运维领域的关系三元组抽取专家。",
            'en': "You are an expert in extracting relational triples from Chinese network operation and maintenance corpus. ",
        },
        'part_task_desc1': {
            'ch': "我会给你一个网络运维领域的句子，你要找出句中所有具有",
            'en': "I will provide you with a sentence in the field of network operation and maintenance. Your task is to identify all relatinal triples within the sentence that match the relations in the list:",
        },
        'part_task_desc2': {
            'ch': "关系的三元组，",
            'en': ". ",
        },
        'part_ans_form': {
            'ch': "然后按主体、客体在句中的顺序，以 \"(s)主体(r)关系(o)客体(e), ...\" 的格式回答，若一个三元组都没有则回答 \"none\" 。",
            'en': "Then, respond in the format \"s) Subject r) Relation o) Object e) ...\". If no triple is found, reply with \"none\". ",
        },
        'part_expectation': {
            'ch': "关系词必须从上面的关系列表中选择；主体客体都是句中的文本，要找全、找准。",
            'en': "Ensure that both subjects and objects are accurately identified and comprehensive from the sentence. ",
        },
        'part_task_start': {
            'ch': "下面完成实际任务，句子为:",
            'en': "Now, let's proceed with the actual task. The sentence is: ",
        },
    }

    def answer_generate(triple_list):
        ans_temp = ""
        triple_str_list = []
        for triple_str_pos in triple_list:
            subj = triple_str_pos['subject']
            rela = triple_str_pos['predicate']
            obj = triple_str_pos['object']
            triple_str = (subj, rela, obj)
            if triple_str not in triple_str_list:
                # 文本部分完全相同的三元组只取一个。  三元组在文中的位置在长度限制内
                triple_str_list.append(triple_str)
                ans_temp += f"(s){subj}(r){relation_language_adapt(rela, lang)}(o){obj}(e),"
        if ans_temp == "":
            ans_temp = "none"
            # print("none")
        return ans_temp

    with open(file_name, 'r', encoding='UTF-8') as f:
        samples = json.loads(f.read())
        assert type(samples) == list

    with open("dataset/CMIM23-NOM1-RA/train_data.json", 'r', encoding='UTF-8') as f:
        examples = json.loads(f.read())

    # 确定要提问的关系列表
    rela_set_str = ""
    for rela in RElATION_LIST:
        rela_set_str += f"{relation_language_adapt(rela, lang)}, "
    rela_set_str = rela_set_str[:-2]  # example: "定义, 特点"

    samples_out = []
    for _ in range(loop_num):
        for d_i in range(len(samples)):
            sent = samples[d_i]['text']
            triple_str_pos_list = samples[d_i]['relation_list']
            # id_, sent, triple_str_pos_list = samples[d_i].copy()

            # ---------------------------------------- 设计指示
            ques = f"{instructions['part1'][lang]}" \
                          f"{instructions['part_task_desc1'][lang]} [{rela_set_str}] {instructions['part_task_desc2'][lang]}" \
                          f"{instructions['part_ans_form'][lang]}" \
                          f"{instructions['part_expectation'][lang]}" \
                          f"{instructions['part_task_start'][lang]}{sent}"

            # ans
            ans = answer_generate(triple_str_pos_list)

            for _ in range(10):  # 同一个样本重复10遍，每一遍添加不同的oneshot示例
                example_ok = False
                example_str = None
                while example_ok is False:
                    # -------------------- 选择合适的示例
                    example = random.sample(examples, 1)[0]
                    example_sent = example['text']
                    example_ans = answer_generate(example['relation_list'])

                    if len(example_sent) > 50 or example_sent == sent or len(example_ans) > 9999:
                        continue

                    if lang == 'ch':
                        example_str = f"例如，句子为:{example_sent}，则答案为: {example_ans}。"
                    elif lang == 'en':
                        example_str = f"For example, if the sentence is: {example_sent}, the answer would be: {example_ans}. "

                    example_ok = True

                example_pos = ques.find(instructions['part_task_start'][lang])
                ques_with_example = ques[:example_pos] + example_str + ques[example_pos:]

                sample_info = {
                    "instruction": "",
                    "input": ques_with_example,
                    "output": ans,
                }
                samples_out.append(sample_info)
            # ^^^ for _ in range(10):
        # ^^^ for d_i in range(len(samples)):
    # ^^^ for _ in loop_num:

    return samples_out


def convert__english__rela_order__sample_expand(file_name, loop_num=1):
    return convert__rela_order__sample_expand(file_name, loop_num, lang='en')


def convert__english__rela_order__sample_expand__oneshot(file_name, loop_num=1):
    """
    t5 tokenizer 的token长度统计 粗略结果 为：
        ques_len = [[300, 0.03], [400, 0.85], [500, 0.1], [600, 0.01], [700, 0.0], [800, 0.0], [900, 0.0], [1000, 0.0], [1100, 0.0], [1200, 0.0]]
        ans_len = [[300, 0.61], [400, 0.17], [500, 0.06], [600, 0.0], [700, 0.06], [800, 0.0], [900, 0.06], [1000, 0.0], [1100, 0.06], [1200, 0.0]]
    :param file_name:
    :param loop_num:
    :return:
    """
    return convert__rela_order__sample_expand__oneshot(file_name, loop_num, lang='en')


def convert__english__triple_order__sample_expand__oneshot(file_name, loop_num=1):
    return convert__triple_order__sample_expand__oneshot(file_name, loop_num, lang='en')


def convert_dataset():

    # convert_func = convert__rela_order__sample_expand   # 不同的格式选择
    # output_dir = "dataset/CMIM23-NOM1-RA_llm_form/rela_order__sample_expand"

    # convert_func = convert__rela_order__sample_expand__oneshot   # 不同的格式选择
    # output_dir = "dataset/CMIM23-NOM1-RA_llm_form/rela_order__sample_expand__oneshot"
    # # ^^^ 示例从 convert__rela_order__sample_expand 的 train 中抽取，因此需要先生成 convert__rela_order__sample_expand

    # convert_func = convert__english__rela_order__sample_expand   # 不同的格式选择
    # output_dir = "dataset/CMIM23-NOM1-RA_llm_form/english__rela_order__sample_expand"

    # convert_func = convert__english__rela_order__sample_expand__oneshot   # 不同的格式选择
    # output_dir = "dataset/CMIM23-NOM1-RA_llm_form/english__rela_order__sample_expand__oneshot"
    # # ^^^ 示例从 convert__english__rela_order__sample_expand 的 train 中抽取，因此需要先生成 convert__english__rela_order__sample_expand

    convert_func = convert__triple_order__sample_expand__oneshot   # 不同的格式选择
    output_dir = "dataset/CMIM23-NOM1-RA_llm_form/triple_order__sample_expand__oneshot"
    # ^^^ 示例从 原数据集 的 train 中抽取，因此可直接运行

    # convert_func = convert__english__triple_order__sample_expand__oneshot   # 不同的格式选择
    # output_dir = "dataset/CMIM23-NOM1-RA_llm_form/english__triple_order__sample_expand__oneshot"
    # # ^^^ 示例从 原数据集 的 train 中抽取，因此可直接运行

    if os.path.exists(output_dir) == 0:
        os.makedirs(output_dir)

    print("\ndev")
    random.seed(240922)    # 设置种子，并重置随机数
    output = convert_func("dataset/CMIM23-NOM1-RA/valid_data.json")
    length_statistic(output)
    print(f"len(output) = {len(output)}")
    with open(os.path.join(output_dir, "dev.json"), "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=4)

    print("\ntest")
    random.seed(240923)    # 设置种子，并重置随机数
    output = convert_func("dataset/CMIM23-NOM1-RA/test_data.json")
    length_statistic(output)
    print(f"len(output) = {len(output)}")
    with open(os.path.join(output_dir, "test.json"), "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=4)

    print("\ntrain")
    random.seed(240921)    # 设置种子，并重置随机数
    output = convert_func("dataset/CMIM23-NOM1-RA/train_data.json", 5)
    length_statistic(output)
    print(f"len(output) = {len(output)}")
    with open(os.path.join(output_dir, "train.json"), "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=4)

    print("\nexamples")
    with open(os.path.join(output_dir, "examples.json"), "w", encoding="utf-8") as fp:
        json.dump(output[:100], fp, ensure_ascii=False, indent=4)


def json_key_justify():
    """
    细微调整 json 文件不同的格式。
    针对具体实例具体调整
    :return:
    """

    input_dir = "dataset/CMIM23-NOM1-RA_llm_form/triple_order__sample_expand__oneshot"
    output_dir = "dataset/CMIM23-NOM1-RA_llm_form/temp"

    file_list = list(os.walk(input_dir))[0][-1]
    print(file_list)
    for file_name in file_list:
        if file_name[-5:] != ".json":
            continue
        with open(os.path.join(input_dir, file_name), 'r', encoding='UTF-8') as f:
            samples = json.loads(f.read())
        samples_out = []
        for sample in samples:
            ques = sample['input']
            ans = sample['output']
            samples_out.append({
                    "conversations": [
                        {
                            "role": "user",
                            "content": ques
                        },
                        {
                            "role": "assistant",
                            "content": ans
                        }
                    ]
            })
        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as fp:
            json.dump(samples_out, fp, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    convert_dataset()

    # json_key_justify()

