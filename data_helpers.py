import math
import pyarrow as pa
import pyarrow.compute as compute
import random
import torch


from dataclasses import dataclass
from datasets import Dataset
from transformers import BatchEncoding
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Dict


'''
    This is the main class where we implement all our customizations.
    We implement a custom data collator to handle dynamic MLM and NSP.
    We have access to the entire dataset text during this time. 
'''
@dataclass
class DataCollatorForPretraining(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    dataset: Dataset
    doc_lens: Dict[str, int]
    max_segment_size: int = 510
    sample_pairs: bool = False
    mlm_prob: float = 0.15
    return_tensors: str = 'pt'

    '''
        This function chooses 15% tokens for masking in a way that ensures whole-word masking
    '''
    def get_mask_candidates(self, input_ids, attention_mask, special_tokens_mask, split_token_prefix='##'):
        valid_tokens_mask = attention_mask & ~special_tokens_mask
        lm_mask = torch.zeros_like(input_ids).bool()
        
        # each candidate is a set of tokens that form a whole word
        for i, sent in enumerate(input_ids):
            sent_len = attention_mask[i].float().sum().item()
            if sent_len == 0: continue
            
            cand_indices = []
            for k, token in enumerate(sent):
                if not valid_tokens_mask[i, k]: continue
                if len(cand_indices) > 0 and self.tokenizer._convert_id_to_token(token).startswith(split_token_prefix): #or not tokenizer._convert_id_to_token(token).startswith(space_prefix)):
                    cand_indices[-1].append(k)
                else: cand_indices.append([k])
                
            # this step actually chooses the random candidates
            random.shuffle(cand_indices)
            max_cands = min(max(1, int(round(sent_len * self.mlm_prob))), 512)
            mask_indices = []
            for idx_set in cand_indices:
                if len(mask_indices) + len(idx_set) > max_cands:
                    continue
                mask_indices.extend(idx_set)

            lm_mask[i, mask_indices] = True

        return lm_mask

    '''
        This function creates NSP pairs.
        The 'input_ids' passed to this function are retained as 'anchor_input_ids'
        Their respective NSP pairs are created and added to 'pair_input_ids'
    '''
    def sample_pairs_for_nsp(self, examples):
        example_pairs = []
        
        # iterate over all blocks
        for i, doc in enumerate(examples):
            # extract doc_id and block_id
            anchor_id = doc['sample_ids']
            doc_id, block_id = anchor_id.split(':')
            block_id = int(block_id)
            
            
            if doc_id in self.doc_lens and block_id < self.doc_lens[doc_id] - 1:
                anchor = doc['input_ids']
                sampling_choice = random.randint(0,1)   # whether to pair with positive (0) or negative (1)
                if sampling_choice == 0:
                    pos_pair_id = f'{doc_id}:{block_id + 1}'
                    
                    # efficient searching since dataset is based on PyArrow backend
                    pos_flag = compute.equal(self.dataset.data['sample_ids'], pos_pair_id)
                    pos_pair = self.dataset.data.filter(pos_flag).to_pydict()['input_ids']
                    if len(pos_pair) == 0:  # next sent not found
                        continue
                    pair = torch.tensor(pos_pair[0], dtype=torch.long)
                    pair_id, sr_label = pos_pair_id, 0
                else:
                    neg_pair_candidates = torch.randperm(self.doc_lens[doc_id])
                    neg_pair_id = None
                    
                    # this and next sent cannot be a negative example, so keep searching till you get it
                    for cand in neg_pair_candidates:
                        if cand not in [block_id, block_id + 1]:
                            neg_pair_id = f'{doc_id}:{cand}'
                            break
                    if neg_pair_id is None:
                        continue
                    
                    # efficient searching since dataset is based on PyArrow backend
                    neg_flag = compute.equal(self.dataset.data['sample_ids'], neg_pair_id)
                    neg_pair = self.dataset.data.filter(neg_flag).to_pydict()['input_ids']
                    if len(neg_pair) == 0:
                        continue
                    pair = torch.tensor(neg_pair[0], dtype=torch.long)
                    pair_id, sr_label = neg_pair_id, 1
                example = {'sample_ids': f'{anchor_id}/{pair_id}', 'anchor_input_ids': anchor, 'pair_input_ids': pair, 'sr_labels': sr_label}
                example_pairs.append(example)
        return example_pairs

    '''    
        The memory-mapped Dataset is maintained as Dict[List], where each Dict key corresponds to each field.
        When passed through a dataloader with batch size k, it extracts k-sized List[Dict], where each Dict key corresponds to each field.
        This is passed as input to any data collator function.
        For our custom data collator, we need to implement the torch_call function, which takes as input the k-sized List[Dict]
    '''
    def torch_call(self, examples):

        # first call the NSP function, which replaces the 'input_ids' element with 'anchor_input_ids' and 'pair_input_ids'
        # it also adds a new key 'sr_labels' for NSP labels
        if self.sample_pairs:   
            examples = self.sample_pairs_for_nsp(examples) 
        
        # token_type_ids needed since we are doing NSP
        input_ids = torch.full((len(examples), self.max_segment_size + 2), self.tokenizer.pad_token_id, dtype=torch.long)
        token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long)

        for i, doc in enumerate(examples):
            if 'input_ids' in doc:
                sent = doc['input_ids'][:self.max_segment_size]
                input_ids[i, 1:len(sent) + 1] = sent
                input_ids[i, 0] = self.tokenizer.cls_token_id
                input_ids[i, len(sent) + 1] = self.tokenizer.sep_token_id
            
            else:
                anchor = doc['anchor_input_ids']
                pair = doc['pair_input_ids']
                
                # when len(anchor) + len(pair) > max_segment_size, we need a truncation strategy
                # here we first remove excess tokens from the longer string
                # then we keep removing same no. of tokens from each string 
                if len(anchor) + len(pair) > self.max_segment_size - 1:
                    smaller_len = min(len(anchor), len(pair))
                    diff = self.max_segment_size - 1 - 2 * smaller_len
                    if diff >= 0:
                        if len(anchor) > len(pair): anchor = anchor[:smaller_len + diff]
                        else: pair = pair[:smaller_len + diff]
                    else:
                        anchor = anchor[:math.ceil((self.max_segment_size - 1) / 2)]
                        pair = pair[:math.floor((self.max_segment_size - 1) / 2)]
                
                # place the anchor, pair and special tokens in appropriate position        
                input_ids[i, 1:len(anchor) + 1] = anchor
                input_ids[i, len(anchor) + 2:len(pair) + len(anchor) + 2] = pair
                input_ids[i, 0] = self.tokenizer.cls_token_id
                input_ids[i, [len(anchor) + 1, len(anchor) + len(pair) + 2]] = self.tokenizer.sep_token_id
                token_type_ids[i, len(anchor) + 1 : len(anchor) + len(pair) + 3] = 1
            
        attention_mask = input_ids != self.tokenizer.pad_token_id
        special_tokens_mask = (input_ids == self.tokenizer.cls_token_id) + (input_ids == self.tokenizer.sep_token_id) + (input_ids == self.tokenizer.pad_token_id)

        # get the mask for indicating MLM candidate tokens, and create MLM labels
        lm_mask = self.get_mask_candidates(input_ids, attention_mask, special_tokens_mask)
        labels = input_ids.clone()
        labels[~lm_mask] = -100

        # replace the original tokens with [MASK] 80% time
        final_lm_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & lm_mask
        input_ids[final_lm_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # replace the original tokens with random tokens 10% time, unchanged for the rest
        random_mask = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & lm_mask & ~final_lm_mask
        random_tokens = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        input_ids[random_mask] = random_tokens[random_mask]

        batch = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask, 'mlm_labels': labels}
        if 'sr_labels' in examples[0]:
            batch['sr_labels'] = torch.tensor([e['sr_labels'] for e in examples])

        return BatchEncoding(batch)

'''
    Use this function if you want equal-sized chunks/blocks.
    Dataset doc text may already be divided into sentences, or a single string.
'''
def chunk_text_blocks(examples, block_size=254):
    sample_ids = []     # string identifier
    input_ids = []      # convert list of sentences string into list of equal-sized chunks/blocks
    
    # iterate over documents
    for i in range(len(examples['input_ids'])):     
        sample_id = examples['id'][i]
        block_id = 0
        block_input_ids = []
        
        # iterate over sentences in i-th doc 
        for j in range(len(examples['input_ids'][i])):
            
            # iterate over tokens in i-th doc j-th sent
            for k in range(len(examples['input_ids'][i][j])):
                
                # current block is now full, dump it to inputs 
                if len(block_input_ids) == block_size:
                    sample_ids.append(f'{sample_id}:{block_id}')
                    input_ids.append(block_input_ids)
                    block_id += 1
                    block_input_ids = []
                
                # add current token to current block   
                block_input_ids.append(examples['input_ids'][i][j][k])
                
        # add remaining tokens in doc, this block will have size < block_size
        if len(block_input_ids) > 0:
            sample_ids.append(f'{sample_id}:{block_id}')
            input_ids.append(block_input_ids)
                
    return {'sample_ids': sample_ids, 'input_ids': input_ids}

'''
    Use this function if you want to keep real sentences and not chunks.
    Dataset doc text must be already divided into sentences.
'''
def chunk_text_line_by_line(examples): 
    sample_ids = [] 
    input_ids = []
    for i in range(len(examples['input_ids'])):
        curr_sample_id = examples['id'][i]
        sample_ids += [f'{curr_sample_id}:{j}' for j in range(len(examples['input_ids'][i]))]
        input_ids += examples['input_ids'][i]
    return {'sample_ids': sample_ids, 'input_ids': input_ids}

'''
    Used for calculating the number of chunks/sentences per document --- needed for NSP
'''
def get_doc_lens(example, dataset=None):
    table = dataset.data
    flag = compute.equal(pa.array([sid.as_py().split(':')[0] for sid in table['sample_ids']]), example['id'])
    return {'doc_lens': len(table.filter(flag))}