""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
from .utils_ner import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, input_len,segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# import binascii
# from Crypto.Cipher import DES

t1_sent_part_li = [(b'a2028ae3377bc4d0787c572c57277e43b3f778feab1cfe3b7efb9ce9138554c4a8cdd7c3ed09ee39918c4d7bc272a239a7aeda3aafc52622088b63c6e668471f1226', b'1fe5878bae7d8545bd29544157864cf2afaece23467900f33a8000023e4a5d66232b0306c56457dba0931bf92b06bfc240ea64766abf00960586d5495bdafc336c0e422fde43eda78e8bb0b97af12bc6d97c3c4a4ecc31c172121d38291951e413fb4cda5a8c0e0766d6de2c171bfa78a029a84d40f0beafef8720dd357cef7c952b4b45ad34efd5ff8da1c43dc9fe1e440d1be415e2d9ca79bec14213dbe8a9b94370249a963e9fad27d5b92a78b0b18c9873d6be41b216935a469e70302fe0a98bc4f89c6d1f9cbb3b227a013bc94571f8559a8bf89d0ead1a')]
t1_ent_li = [(b'02eb79242451be107d', b'02eb792214750c4bb4'), (b'a204fe7f1b6d19216d23c711', b'a204fe7f1b6d19216d23c7115514b9bef368'), (b'a106ab1ca0d7a974d58d59e36016d32f03ab70c5cde6123c41a82e815b17c6d423a8eba2fd9ca0601bace994335b679c4181f0671a197bc0', b'114694b08a9b1b65f83298343e52d007e2cf9b85'), (b'ad77b7d21231946d54', b'ad77b7d21231946d54f0b23585a6e5'), (b'a0bf5ce6e39c2a49d7542b6976cc0a', b'a0bf578ee7c61cfff74d4183'), (b'a135192af98a772ccac8af2af209ae33f6526bfe20f09092775dfd2ab52eed22fad93d1c35f8a461ca3107a2b52075b8d9482208004eca380e9b', b'a135192af98a772ccac8af2af209ae33f6526bfe20f09092775dfd2ab52eed22fad93d1c35f8a461ca3107a45f2a4e54491246936e74e4eee75d'),
             (b'a2207a36630bb23141dabbd4283f89042c0ef51606e267b4f1251a0bd9e6d3d11f4b76aec6b3aa34e44e5b804399df13f18e1d6c63e213', b'a2207a36630bb23141dabbd4283f89042c0ef51606e267b4f1251a0bd9e6d3d11f4b76aec6b3aa34e44e5d686515998b1b6517c057526f'), (b'0aea1073cf01007b75', b'0aea1075197cfc6c3c'), (b'a120bcffb24077698de4dc483a1211b3c785026b83aee6322a72a0e7fb46ee74f4ed68bb9891704b654d04bcd69e10defdb056a1b6d17cfd3534f0f48a7cd8e95e0bd194f0', b'a120bcffb24077698de4dc483a1211b3c785026b83aee6322a72a0e7fb46ee74f4ed68bb9891704b654d04bcd69e10defbb8f365d29a530c318ff323cfb20abf565ee1405d'), (b'0b29d00cd516a57323e45f', b'0b29d00cd328259513c374'),
             (b'14fe37c4d987a67242a262764529f37d', b'14fe37c4d987a6742e8ca688062c5690'), (b'0d884edef874401dd29a03df0cf480897fface108f', b'0d884edef874401dd29a0588996f785901049c7605'), (b'a0b80d1aee4e66200a063e0421935969a23f75630bbdaee168920c8392d71d0aaefb8d7fee57369a24dd0d7416150735d49d586d61f54ed468b29ab3a49fab774567', b'a0b80d1aee4e66200a063e0421935969a23f75630bbdaee168920c8392d71d0aaefb8d7fee57369a24dd0d7416150735d49d586d61f54ed468b29ab3a49fab774567f57dcad0ad7617d98027c12ca359a285e525c08cc6a54067dfa2b2976ca5645f5197dd419feff3d40a81ff8f75ec9d0e610e225587ccd696bc498981'),
             (b'a0bf461ae51d77d49c8a7cbe953006e656b1ef5c489a998a773f86428a8e1a4ab63679760829696eb4f77d47245a1516bef4d14f192859677a5c3ac5278123c8e6197c72a9', b'a0bf461ae51d77d49c8a7cbe953006e656b1ef5c489a998a773f86428a8e1a4ab63679760829696eb4f77d47245a1516bef4d14f19285f6f5fb335a3a36985937702b33b4c'), (b'a21ad423f0ec562b938744baf80ca5154993297841e8714792415739b052', b'a21ad423f0ec562b938744baf80ca513f907f5a6eee112b9b7d18c8ad56a'), (b'17d14aba52ebcd848888a3ab6d09334da65c23d0', b'17d14aba52ebcd848888a3ab6be0976240f6ad2f'),
             (b'ad577474aa5c7865211958ddff9418fe88f166c1f17669c4476bd1b4e977ea19a2fa67ac5b55f95b2fee495873', b'ad577474aa5c7865211958ddff9418fe88f166c1f17669c4476bd1b4e977ea19a2fc8d4d94b52c0755321a07c0'), (b'ace67e610b4cd140722d94a09e03a16cd80b8e6da1f0a57c151d7b297a81f6f15f5485ea230f2616c77b8ad4d7fce89ec59fb4f62b4afab3fcdfa561eca718991d3a4fddeee2ca8433fc26', b'ace67e610b4cd140722d94a09e03a16cd80b8e6da1f0a57c151d7b297a81f6f15f5485ea230f2616c77b8ad4d7fce89ec59fb4f62b4afab3fcd9c26451e28aaae549b4a9d3138b53d0a1c6'), (b'a20713a0a452a85f09a9b1ed23593cbf8b93a5ac976b70cd', b'a20713a0a452a85f09a9b1ed23593cb9407c803017a7ee4c'),
             (b'75246ecfdd6c4d2d69f3d9746e0f6fc1d8a6d156a5939e5645fe848fea5b0b6bb840660f27afaa6190959f7eb8c9e10b5f7b9395e0aa1a82b00620350b2b3484d1cdb2432998f20abaaaeb6de2', b'75246ecfdd6c4d2d69f3d9746e0f6fc1d8a6d156a5939e5645fe848fea5b0b'), (b'055531171907a0cb9a452be8177dc850699843240a2d41b52599e4b18fe1', b'055531171907a0cb9a452be8177dc850699843240a2d47201661083e472f'), (b'a106b1a963318335d9248433984ba37e96348f674a8be12c4abcf46ae00998660e64dc1df238265439772cb454cad4a6dcd8d1f5ba498b4f0ba9afbd7a19dd', b'6c3a458adb4288905a7c14c61b3dc78c9f1cd26a7cc74bd0061e2e9b4c702db043cb02d96200c2abd5d400bd31f649bcba539ccfc8283dcb8379daa9b0b4ee19'), (b'77cce799fe0d3aeeb29b09c9', b'77cce793beb4758cdb48c563')]


def des_encrypt(data, key=b'by640123', iv=b'by640123'):
    cipher = DES.new(key, DES.MODE_CFB, iv)
    data = cipher.encrypt(data.encode())
    return binascii.b2a_hex(data)


def des_decrypt(data, key=b'by640123', iv=b'by640123'):
    data = binascii.a2b_hex(data)
    decipher = DES.new(key, DES.MODE_CFB, iv)
    return decipher.decrypt(data).decode()


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens


def collate_fn_2310ForTask1(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_tag_mask, all_real_sent_ids = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()    # 最长长度取batch中的最大值，而无需都取设定的limit
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    all_labels = all_labels[:,:max_len]
    all_tag_mask = all_tag_mask[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_lens, all_tag_mask, all_real_sent_ids


def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask,input_len = input_len,
                                      segment_ids=segment_ids, label_ids=label_ids))
    return features


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X",'B-CONT','B-EDU','B-LOC','B-NAME','B-ORG','B-PRO','B-RACE','B-TITLE',
                'I-CONT','I-EDU','I-LOC','I-NAME','I-ORG','I-PRO','I-RACE','I-TITLE',
                'O','S-NAME','S-ORG','S-RACE',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CluenerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                'B-organization', 'B-position','B-scene',"I-address",
                "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                'I-organization', 'I-position','I-scene',
                "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                'S-name', 'S-organization', 'S-position',
                'S-scene','O',"[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a= line['words']
            # BIOS
            labels = line['labels']
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

ner_processors = {
    "cner": CnerProcessor,
    'cluener':CluenerProcessor
}
