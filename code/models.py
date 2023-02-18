import torch
from torch.nn import functional, CrossEntropyLoss, Softmax
from torchcrf import CRF
from transformers import RobertaModel, BertModel

from args import args, config
class Model_Crf(torch.nn.Module):
    def __init__(self, config):
        super(Model_Crf, self).__init__()
        self.bert = BertModel.from_pretrained(args.pre_model_name)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, args.label_size)
        self.crf = CRF(num_tags=args.label_size, batch_first=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  context_mask=None, labels=None, span_labels=None, start_positions=None, end_positions=None, testing=False, crf_mask=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output[:,1:-1,:] #remove [CLS], [SEP]
        logits = self.classifier(sequence_output)#[batch, max_len, label_size]
        outputs = (logits,)
        if labels is not None:
            #print('logits = ', logits.size())
            #print('labels = ', labels.size())
            #print('crf_mask = ', crf_mask.size())
            loss = self.crf(emissions = logits, tags=labels, mask = crf_mask, reduction="mean")
            outputs =(-1*loss,)+outputs
        return outputs

class Model_Softmax(torch.nn.Module):
    def __init__(self, config):
        super(Model_Softmax, self).__init__()
        self.bert = BertModel.from_pretrained(args.pre_model_name)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, args.label_size)
        self.loss_calculater = CrossEntropyLoss()
        self.softmax = Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,  context_mask=None, labels=None, span_labels=None, start_positions=None, end_positions=None, testing=False, crf_mask=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output[:,1:-1,:] #remove [CLS], [SEP]
        logits = self.classifier(sequence_output)#[batch, max_len, label_size]
        logits = self.softmax(logits)
        outputs = (logits,)
        if labels is not None:
            #print('logits = ', logits.size())
            #print('labels = ', labels.size())
            labels = functional.one_hot(labels, num_classes=args.label_size).float()
            loss = self.loss_calculater(logits, labels)
            outputs =(loss,)+outputs
        return outputs