import torch
import torch.nn as nn

from transformers import BertModel
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from typing import Optional


'''
    The main loss argument is used by the Trainer for optimization.
    But, for calculating perplexity, we need just the MLM loss.
    Since this is expensive to calculate twice, we return the MLM loss directly too(instead of MLM prediction logits).
    sr_logits is for NSP. 
'''
class InLegalBertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    sr_logits: torch.FloatTensor = None
    mlm_loss: Optional[torch.FloatTensor] = None

'''
    Followed similar structure as BertForPreTraining, except for returning MLM loss instead of prediction logits
'''
class InLegalBertForPreTraining(nn.Module):
    def __init__(self, config, bert=None):
        super().__init__()
        
        self.config = config
        self.bert = BertModel(config) if bert is None else bert
        self.cls = BertPreTrainingHeads(config)


    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, mlm_labels=None, sr_labels=None):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction_logits, seq_relationship_logits = self.cls(bert_outputs.last_hidden_state, bert_outputs.pooler_output)
        
        total_loss, mlm_loss = None, None
        if mlm_labels is not None and sr_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(prediction_logits.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            sr_loss = loss_fct(seq_relationship_logits.view(-1, 2), sr_labels.view(-1))
            total_loss = mlm_loss + sr_loss

        return InLegalBertForPreTrainingOutput(loss=total_loss, sr_logits=seq_relationship_logits, mlm_loss=mlm_loss)
