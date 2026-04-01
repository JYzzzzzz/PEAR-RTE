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

        self.bert = BertModel(config)
        self.dropout_bert_out = nn.Dropout(config.hidden_dropout_prob)

        # self.lstm = nn.LSTM(input_size=config.hidden_size + lstm_input_other_emb_size,
        #                     hidden_size=other_args.lstm_hidden_size,
        #                     num_layers=other_args.lstm_num_layers,
        #                     bidirectional=other_args.lstm_bidirectional, batch_first=True,
        #                     dropout=config.hidden_dropout_prob)

        # self.dropout_lstm_output = nn.Dropout(config.hidden_dropout_prob)

        lstm_output_size = other_args.lstm_hidden_size
        if other_args.lstm_bidirectional is True:  # 若 bidirectional 设为 True ，LSTM 输出维度乘 2
            if independent_bw_lstm[0] is False:   # 正反共用LSTM
                lstm_output_size = other_args.lstm_hidden_size * 2
            else:                           # 正反LSTM独立
                lstm_output_size = other_args.lstm_hidden_size + independent_bw_lstm[1]

        self.classifier = nn.Linear(lstm_output_size, config.num_labels)
        if self.loss_type in ['crf']:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    # def forward(self,
    #             input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None,
    #             tag_mask=None):
    def forward(self, batch, **kwargs):


        # bert
        # bert_input_other_embeddings=None
        outputs = self.bert(input_ids=batch['input_ids'], attention_mask=batch['bert_att_mask'],
                            token_type_ids=batch['segment_ids'],
                            other_embeddings=None)
        # ^^^ outputs = sequence_output, pooled_output, (hidden_states), (attentions)
        #             = (sequence_output, pooled_output) in this application
        # ^^^ size:     sequence_output: torch.Size([4, 47, 768])
        #               pooled_output: torch.Size([4, 768])
        
        bert_seq_output = outputs[0]
        bert_seq_output = self.dropout_bert_out(bert_seq_output)    # (B, L, H)
        logits = self.classifier(bert_seq_output)  # torch.Size([B, seq_len, 34])  34==cls_num
        outputs = (logits,)

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
