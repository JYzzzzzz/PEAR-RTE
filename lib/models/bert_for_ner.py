import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.crf import CRF
from .transformers.modeling_bert import BertPreTrainedModel
from .transformers.modeling_bert import BertModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits
from torch.nn import CrossEntropyLoss
from ..losses.focal_loss import FocalLoss
from ..losses.label_smoothing import LabelSmoothingCrossEntropy


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # 创建交叉熵损失函数，softmax包含在其中。
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                input_lens=None, tag_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # ^^^ outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        #             = (sequence_output, pooled_output) in this application
        # size:     sequence_output: torch.Size([4, 47, 768])     pooled_output: torch.Size([4, 768])
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # torch.Size([B, seq_len, 34])  34==cls_num
        # print(XXXXX)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=tag_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, ):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs


class My_BiLSTM(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 dropout,
                 independent_bw_lstm=False,   # 使用独立的反向lstm
                 independent_bw_lstm_hidden_size=0,     # 反向lstm的隐藏层
                 ):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=dropout)
        self.bidirectional = bidirectional

        self.bw_lstm = None
        if self.bidirectional and independent_bw_lstm:
            self.bw_lstm = nn.LSTM(input_size=input_size,
                                   hidden_size=independent_bw_lstm_hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=False,
                                   batch_first=True,
                                   dropout=dropout)

    def forward(self, inputs, masks):
        """

        :param input: size: (B, seq_len, input_size)
        :param mask: size: (B, seq_len)
        :return:
        """

        output_fw, c = self.lstm(inputs)  # output_fw: (B, seq_len, hidden_size)
        if self.bidirectional is False:  # 单向，直接返回
            return output_fw, c

        """
        反向LSTM的策略 !!!
            1、将inputs的batch中的每一个序列中有效信息移到尾部，然后以序列那一维翻转顺序
            2、放入LSTM中，输出output_bw。
            3、再次将output_bw的batch中的每一个序列中有效信息移到尾部，再次翻转
            4、与正向输出拼接。
            (目的是为了消除较短句子尾部pad或其他token的影响)
        """
        if masks is None:
            masks_len = None
        else:
            # 沿着序列长度的维度（第二维）求和，得到每个batch中1的个数，然后将tensor转换为list
            masks_len = torch.sum(masks, dim=1).cpu().tolist()  # list len=B
        inputs_bw = self.reverse_seqs(inputs, masks_len)
        if self.bw_lstm is None:
            output_bw, c = self.lstm(inputs_bw)  # output_bw: (B, seq_len, hidden_size)
        else:
            output_bw, c = self.bw_lstm(inputs_bw)
        output_bw = self.reverse_seqs(output_bw, masks_len)

        output = torch.cat([output_fw, output_bw], dim=-1)  # size: (B, seq_len, 2*hidden_size)
        return output, c

    def reverse_seqs(self, seqs, masks_len=None):
        """
        不是简单的直接顺序反转，而是根据mask，将句首的有用信息平移到句尾，再进行翻转
        :param seqs:
        :param masks_len: None or a list
        :return:
        """
        if masks_len is None:
            return seqs.flip(dims=[1])

        # 初始化一个新的张量B，用于存储结果，其形状与A相同
        seqs_re = torch.zeros_like(seqs)  # 已确认，返回值位于同一设备
        for i, l in enumerate(masks_len):
            # 分割当前行，第一部分是从0到m-1，第二部分是从m到末尾（不包括末尾）
            first_part = seqs[i, :l]
            second_part = seqs_re[i, l:]  # all 0

            # 将第一部分移动到末尾，即与第二部分拼接
            seqs_re[i] = torch.cat((second_part, first_part), dim=0)

        return seqs_re.flip(dims=[1])  # 在第一维上进行翻转并返回


class BertLstmForNer(BertPreTrainedModel):  # jyz add 2407
    def __init__(self, config, other_args):
        super(BertLstmForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.loss_type = other_args.loss_type  # 损失函数类型
        self.bilstm_len = other_args.bilstm_len  # 指示bilstm的长度，仅在使用本人所写的bilstm代码时生效
        self.tag_bias_para = other_args.tag_bias_para  # 是否进行对部分tag的加权(偏心)，目前仅在使用lsr时有效
        self.lsr_eps = other_args.lsr_eps  # lsr的超参数，仅在使用lsr时有效
        if other_args.indep_bw_lstm_h <= 0:
            independent_bw_lstm = (False, -1)
        else:
            independent_bw_lstm = (True, other_args.indep_bw_lstm_h)

        lstm_input_other_emb_size = 0

        # tag embedding & rela embedding, 为了不同的信息融合方式
        self.triple_last_fuse_mode = other_args.triple_last_fuse_mode
        self.triple_last_fuse_position = other_args.triple_last_fuse_position
        self.triple_last_tag_embeddings = None
        if self.triple_last_fuse_mode in ['entity_emb', 'all_emb']:
            if self.triple_last_fuse_position in ['bert_input_add', 'lstm_input_add']:
                triple_last_dim = config.hidden_size
            else:  # in ['lstm_input_cat']:
                triple_last_dim = other_args.triple_last_cat_embedding_dim
                lstm_input_other_emb_size += triple_last_dim
            self.triple_last_tag_embeddings = nn.Embedding(20, triple_last_dim)  # tag num
            for param in self.triple_last_tag_embeddings.parameters():  # 锁定参数不更新
                param.requires_grad = False
        self.rela_embeddings = None
        if self.triple_last_fuse_mode in ['all_emb']:
            if self.triple_last_fuse_position in ['bert_input_add', 'lstm_input_add']:
                rela_dim = config.hidden_size
            else:  # in ['lstm_input_cat']:
                rela_dim = other_args.triple_last_cat_embedding_dim
                lstm_input_other_emb_size += rela_dim
            self.rela_embeddings = nn.Embedding(20, rela_dim)  # rela num

        # tag embedding & rela embedding, 输入lstm时的配套组件
        self.lstm_input_other_emb_layernorm = None
        # self.lstm_input_other_emb_dropout = None
        if self.triple_last_fuse_mode in ['entity_emb', 'all_emb']:
            if self.triple_last_fuse_position in ['lstm_input_add']:
                self.lstm_input_other_emb_layernorm = nn.LayerNorm(config.hidden_size)
                # self.lstm_input_other_emb_dropout = nn.Dropout(config.hidden_dropout_prob)
            elif self.triple_last_fuse_position in ['lstm_input_cat']:
                self.lstm_input_other_emb_layernorm = nn.LayerNorm(lstm_input_other_emb_size)
                # self.lstm_input_other_emb_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.bert = BertModel(config)
        # self.dropout_bert_output = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_lstm_input = nn.Dropout(config.hidden_dropout_prob)

        ########################
        ########## LSTM init
        self.lstm = nn.LSTM(input_size=config.hidden_size + lstm_input_other_emb_size,
                            hidden_size=other_args.lstm_hidden_size,
                            num_layers=other_args.lstm_num_layers,
                            bidirectional=other_args.lstm_bidirectional, batch_first=True,
                            dropout=config.hidden_dropout_prob)
        # self.my_lstm = My_BiLSTM(
        #     input_size=config.hidden_size + lstm_input_other_emb_size,
        #     hidden_size=other_args.lstm_hidden_size,
        #     num_layers=other_args.lstm_num_layers,
        #     bidirectional=other_args.lstm_bidirectional,
        #     dropout=config.hidden_dropout_prob,
        #     independent_bw_lstm=independent_bw_lstm[0],
        #     independent_bw_lstm_hidden_size=independent_bw_lstm[1],
        # )  # batch_first=True [default]

        self.dropout_lstm_output = nn.Dropout(config.hidden_dropout_prob)

        lstm_output_size = other_args.lstm_hidden_size
        if other_args.lstm_bidirectional is True:  # 若 bidirectional 设为 True ，LSTM 输出维度乘 2
            if independent_bw_lstm[0] is False:   # 正反共用LSTM
                lstm_output_size = other_args.lstm_hidden_size * 2
            else:                           # 正反LSTM独立
                lstm_output_size = other_args.lstm_hidden_size + independent_bw_lstm[1]
        # lstm_output_size = config.hidden_size
        ########## LSTM init
        ########################

        self.classifier = nn.Linear(lstm_output_size, config.num_labels)
        if self.loss_type in ['crf']:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    # def forward(self,
    #             input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None,
    #             tag_mask=None):
    def forward(self, batch, print_flag=False):

        if print_flag:
            print("-- batch")
            print(batch)

        # tag embedding & rela embedding, 为了不同的信息融合方式
        triple_last_tag_embeddings = None
        if self.triple_last_fuse_mode in ['entity_emb', 'all_emb']:
            triple_last_tag_embeddings = self.triple_last_tag_embeddings(batch['label_ids_last'])

            if print_flag:
                print("-- triple_last_tag_embeddings")
                print(triple_last_tag_embeddings.size())
                print(triple_last_tag_embeddings)

        rela_embeddings = None
        if self.triple_last_fuse_mode in ['all_emb']:
            seq_len = batch['input_ids'].size(1)
            rela_embeddings = self.rela_embeddings(batch['rela_id'])
            rela_embeddings = torch.repeat_interleave(rela_embeddings.unsqueeze(1), seq_len, dim=1)

        # other embedding for bert input
        bert_input_other_embeddings = None
        if self.triple_last_fuse_mode in ['all_emb'] and self.triple_last_fuse_position in ['bert_input_add']:
            bert_input_other_embeddings = triple_last_tag_embeddings + rela_embeddings
        elif self.triple_last_fuse_mode in ['entity_emb'] and self.triple_last_fuse_position in ['bert_input_add']:
            bert_input_other_embeddings = triple_last_tag_embeddings

        # bert
        # bert_input_other_embeddings=None
        outputs = self.bert(input_ids=batch['input_ids'], attention_mask=batch['bert_att_mask'],
                            token_type_ids=batch['segment_ids'],
                            other_embeddings=bert_input_other_embeddings)
        # ^^^ outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        #             = (sequence_output, pooled_output) in this application
        # ^^^ size:     sequence_output: torch.Size([4, 47, 768])
        #               pooled_output: torch.Size([4, 768])
        bert_seq_output = outputs[0]
        # sequence_output = self.dropout_bert_output(sequence_output)

        if print_flag:
            print("-- bert_seq_output")
            print(bert_seq_output.size())
            print(bert_seq_output)

        # other embeddings for lstm input
        lstm_input = bert_seq_output
        if self.triple_last_fuse_mode in ['all_emb'] and self.triple_last_fuse_position in ['lstm_input_add']:
            lstm_input += triple_last_tag_embeddings + rela_embeddings
        elif self.triple_last_fuse_mode in ['entity_emb'] and self.triple_last_fuse_position in ['lstm_input_add']:
            lstm_input += triple_last_tag_embeddings
        elif self.triple_last_fuse_mode in ['all_emb'] and self.triple_last_fuse_position in ['lstm_input_cat']:
            lstm_input = torch.cat([bert_seq_output, triple_last_tag_embeddings, rela_embeddings], dim=-1)
        elif self.triple_last_fuse_mode in ['entity_emb'] and self.triple_last_fuse_position in ['lstm_input_cat']:
            lstm_input = torch.cat([bert_seq_output, triple_last_tag_embeddings], dim=-1)

        if print_flag:
            print("-- lstm_input")
            print(lstm_input.size())
            print(lstm_input)

        # LSTM
        lstm_input = self.dropout_lstm_input(lstm_input)    # (B, L, H)

        # lstm_seq_output, _ = self.lstm(lstm_input)  # out size: (4, 47, hidden_layer_size * num_directions)
        # # bilstm_mask = None
        # # if self.bilstm_len in ['tag_mask', 'bert_att_mask']:
        # #     bilstm_mask = batch[self.bilstm_len]
        # # lstm_seq_output, _ = self.my_lstm(lstm_input, bilstm_mask)
        #
        # lstm_seq_output = self.dropout_lstm_output(lstm_seq_output)

        lstm_seq_output = lstm_input    # ablation study. no LSTM

        logits = self.classifier(lstm_seq_output)  # torch.Size([B, seq_len, 34])  34==cls_num
        # print(XXXXX)
        outputs = (logits,)

        if print_flag:
            print("-- logits")
            print(logits.size())
            print(torch.argmax(F.softmax(logits, dim=-1), dim=-1))

        if batch['label_ids'] is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce', 'crf']
            if self.loss_type in ['lsr', 'focal', 'ce']:
                # SOFTMAX+CrossEntropy. 创建交叉熵损失函数，softmax包含在其中。
                if self.loss_type == 'lsr':
                    if self.tag_bias_para > 0:
                        loss_fct = LabelSmoothingCrossEntropy(eps=self.lsr_eps, ignore_index=0, reduction='sum')
                    else:
                        loss_fct = LabelSmoothingCrossEntropy(eps=self.lsr_eps, ignore_index=0)
                elif self.loss_type == 'focal':
                    loss_fct = FocalLoss(ignore_index=0)
                else:
                    loss_fct = CrossEntropyLoss(ignore_index=0)
                # Only keep active parts of the loss
                if batch['tag_mask'] is not None:
                    # active_loss = tag_mask.view(-1) == 1
                    # active_logits = logits.view(-1, self.num_labels)[active_loss]
                    # active_labels = labels.view(-1)[active_loss]
                    active_loss = batch['tag_mask'].reshape(-1) == 1

                    active_logits = logits.reshape(-1, self.num_labels)[active_loss]
                    # ^^^ logits.size() = torch.Size([B, seq_len, tag_num])
                    # ^^^ active_logits.size() = torch.Size([B*seq_len - <tag_mask num>, tag_num])
                    active_labels = batch['label_ids'].reshape(-1)[active_loss]
                    # ^^^ batch['label_ids'].size() = torch.Size([B, seq_len])
                    # ^^^ active_labels.size() = torch.Size([B*seq_len - <tag_mask num>])
                    if self.tag_bias_para > 0:
                        nobias_tag_index = active_labels == 1  # 这里的1表示tag中的‘O（other）’
                        loss_nobias_tag = loss_fct(active_logits[nobias_tag_index], active_labels[nobias_tag_index]) if \
                            True in nobias_tag_index else 0
                        bias_tag_index = active_labels != 1  # 这里的1表示tag中的‘O（other）’
                        loss_bias_tag = loss_fct(active_logits[bias_tag_index], active_labels[bias_tag_index]) if \
                            True in bias_tag_index else 0
                        loss = (loss_nobias_tag + loss_bias_tag * self.tag_bias_para) / active_labels.size()[0]
                    else:
                        loss = loss_fct(active_logits, active_labels)

                else:
                    # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                    loss = loss_fct(logits.reshape(-1, self.num_labels), batch['label_ids'].reshape(-1))
                outputs = (loss,) + outputs
            elif self.loss_type in ['crf']:
                # CRF
                loss = self.crf(emissions=logits, tags=batch['label_ids'], mask=batch['tag_mask'])
                outputs = (-1 * loss,) + outputs

        return outputs  # (loss), scores

    def logits_decode(self, logits, mask=None):
        if self.loss_type in ['lsr', 'focal', 'ce']:
            probs = F.softmax(logits, dim=-1)
            tags = torch.argmax(probs, dim=-1)
        elif self.loss_type in ['crf']:
            tags = self.crf.decode(logits, mask)

        return tags
