

def get_longest_same_str(str1_in, str2_in, start_len=-1):
    """
    找最长子串
    :param start_len: 设定从长度为多少开始，用于多次查找多段相同子串。取-1时表示不限起始长度
    :return: list
    """
    # 判断两个字符串长短，取短的那个赋给str1，进行操作
    str_short, str_long = str1_in, str2_in
    if len(str1_in) > len(str2_in):
        str_short, str_long = str2_in, str1_in
    res_list = []    # 用列表来接收最终的结果，以免出现同时有多个相同长度子串被漏查的情况
    pos_in_str1_list = []
    pos_in_str2_list = []
    # 从str1全长开始进行检测，逐渐检测到只有1位
    start_len = len(str_short) if start_len < 0 or start_len > len(str_short) else start_len
    for r in range(start_len, 0, -1):
        j = 0
        while r <= len(str_short):
            str_short_sub = str_short[j:r]
            if str_short_sub in str_long:
                res_list.append(str_short_sub)
                pos_in_str1_list.append(str1_in.find(str_short_sub))
                pos_in_str2_list.append(str2_in.find(str_short_sub))
            r += 1
            j += 1
        # 判断当前长度下，是否存在子串
        if len(res_list) > 0:
            return res_list, pos_in_str1_list, pos_in_str2_list
    return res_list, pos_in_str1_list, pos_in_str2_list   # []


def str_similarity(str1_in, str2_in, section=2,
                   pos_in_str1_mode="l2r", pos_in_str2_mode="l2r",
                   output_order_mode="long2short"):
    """
    比较两字符串的相似度。
        具体的比较过程为，寻找str1与str2中 最长的、第二长、...的 相同子串str_sub1, str_sub2, ...
        找不到可以为''，str_sub1, str_sub2, ...在str1、str2中的位置必须都没有重叠
    :param: section: 可以有几段
            pos_in_str1_mode, pos_in_str2_mode :
                'l2r': 在str1（str2）中的位置先找最靠左的（从前往后）
                'r2l': 从后往前找
            output_order_mode: 输出子串列表的顺序
                'long2short': 从长到短
                'str1': 按在str1中的顺序
                'str2': 按在str2中的顺序
    :return: 返回4项：
            1、str_sub_list
            1、simi1 = len(str_sub1+str_sub2+...) / len(str1)
            2、simi2 = len(str_sub1+str_sub2+...) / len(str2)
            3、simi_avr = 2*simi1*simi2 / (simi1+simi2)
    :version:
        240304: 添加 output_order_mode
    """
    str1, str2 = str1_in, str2_in

    # 字符串为空的特殊情况
    if str1 == '':
        return [], 0, 0., 0., [], []
    if str2 == '':
        return [], 0, 0., 0., [], []

    # 确定替换字符
    pad_char_list = ['*', '@', '#', '%', '&', '-', '=']
    i = 0
    str1_pad = pad_char_list[i]
    while str1_pad in str1+str2:  # 满足条件则删除该字符，继续挑选
        i += 1
        assert i < len(pad_char_list), "无法确定填充字符\n{}".format(str1)
        str1_pad = pad_char_list[i]
    i = 0
    str2_pad = pad_char_list[i]
    while str2_pad in str1+str2 or str1_pad == str2_pad:
        i += 1
        assert i < len(pad_char_list), "无法确定填充字符\n{}".format(str2)
        str2_pad = pad_char_list[i]

    # 查找子串
    # str_sub_list = [''] * section   # ['', '']
    str_sub_list = []
    str_sub_pos_in_str1 = []
    str_sub_pos_in_str2 = []
    res_l = -1
    for i in range(section):
        if str1 == str1_pad*len(str1) or str2 == str2_pad*len(str2):
            break
        res, pos1_list, pos2_list = get_longest_same_str(str1, str2, res_l)  # res_l设定子串起始最长长度，缩短查找时间
        if not res:  # res==[]
            break
        sub = res[0]
        sub_l = len(res[0])
        pos_in_str1 = pos1_list[0]
        if pos_in_str1_mode == 'r2l':
            while str1.find(sub, pos_in_str1+sub_l) >= 0:
                pos_in_str1 = str1.find(sub, pos_in_str1+sub_l)
        pos_in_str2 = pos2_list[0]
        if pos_in_str2_mode == 'r2l':
            while str2.find(sub, pos_in_str2+sub_l) >= 0:
                pos_in_str2 = str2.find(sub, pos_in_str2+sub_l)
        str1 = str1[:pos_in_str1] + str1[pos_in_str1:].replace(sub, str1_pad*sub_l, 1)
        # str1 = str1.replace(res[0], str1_pad*sub_l, 1)
        assert str1_pad*sub_l in str1
        str2 = str2[:pos_in_str2] + str2[pos_in_str2:].replace(sub, str2_pad*sub_l, 1)
        assert str2_pad*sub_l in str2
        # print(str1)
        # print(str2)
        str_sub_list.append(sub)
        str_sub_pos_in_str1.append(pos_in_str1)
        str_sub_pos_in_str2.append(pos_in_str2)

    # 计算相似度
    str_sub_len = 0
    for s in str_sub_list:
        str_sub_len += len(s)
    simi1 = str_sub_len / len(str1)
    simi2 = str_sub_len / len(str2)
    simi_avr = 2 * simi1 * simi2 / (simi1 + simi2) if (simi1 + simi2) > 0 else 0

    # 列表顺序调整
    if output_order_mode != "long2short" and len(str_sub_list) > 1:
        list_temp = []
        for i in range(len(str_sub_list)):
            list_temp.append([str_sub_list[i], str_sub_pos_in_str1[i], str_sub_pos_in_str2[i]].copy())

        if output_order_mode == "str1":
            list_temp.sort(key=lambda x: x[1])
        elif output_order_mode == "str2":
            list_temp.sort(key=lambda x: x[2])

        for i in range(len(list_temp)):
            str_sub_list[i] = list_temp[i][0]
            str_sub_pos_in_str1[i] = list_temp[i][1]
            str_sub_pos_in_str2[i] = list_temp[i][2]

    return str_sub_list, simi1, simi2, simi_avr, str_sub_pos_in_str1, str_sub_pos_in_str2
        # """
        #   输入 str1 = LTETDD平均RTT
        #       str2 = 使用HARQ技术时，LTETDD使用的控制信令比LTEFDD更复杂，且平均RTT稍长于LTEFDD的8ms。
        # 输出：
        #     str_sub_list = ['LTETDD', '平均RTT']
        #     simi1、simi2、simi_avr ： 相似度指标
        #     str_sub_pos_in_str1 = [0, 6] 表示子串在str1中的位置
        #     str_sub_pos_in_str2 = [10, 35]  表示子串在str2中的位置
        # """


def find_all_pos_in_sent(sent, str1, find_range=None):
    # 查找字符串在文中一定范围内所有切片位置, 返回[(n1,n2), (n3,n4), ...]
    # str1本身不再分段
    str_pos_list = []
    if find_range is None:
        str_find_range = [0, len(sent)]
    else:
        str_find_range = find_range.copy()  # [0, len(sent)]
    while sent.find(str1, str_find_range[0], str_find_range[1]) >= 0:
        pos_start = sent.find(str1, str_find_range[0], str_find_range[1])
        t_pos = (pos_start, pos_start + len(str1))
        assert sent[t_pos[0]:t_pos[1]] == str1, f"{sent}\n{str1}\n{sent[t_pos[0]:t_pos[1]]}"
        str_pos_list.append(t_pos)
        str_find_range[0] = t_pos[1]
    return str_pos_list


p_get = ['./lib/tools/t2db',]


if __name__ == "__main__":
    str1 = 'BCCH电平值【邻区A与B的BCCH的接收电平混合值'
    str2 = '某手机有2个相同BCCH的邻区A和B，但BSIC不同。邻区列表中，此BCCH频点有相应的一个电平值，BSIC显示为邻区A的值。所测到的该BCCH上的电平值应该是什么?'
    # print(str_similarity(str1, str2, section=10, pos_in_str2_mode='r2l'))
    # a, b = 0.5, 0.7
    # print(2*a*b/(a+b))
    str1 = "abcdefg1234567"
    str2 = "12abc"
    print(str_similarity(str1, str2, section=10, output_order_mode="str2"))


