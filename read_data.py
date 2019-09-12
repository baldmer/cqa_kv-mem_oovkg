""" based on the implementation of https://github.com/amritasaha1812/CSQA_Code/blob/master/read_data.py """

import sys
import numpy as np
import pickle as pkl
import os
start_symbol_index = 0
end_symbol_index = 1
unk_symbol_index = 2
pad_symbol_index = 3
kb_pad_idx = 0
nkb = 1
import os


def get_utter_seq_len(dialogue_dict_w2v, dialogue_dict_kb, dialogue_target, dialogue_sources, dialogue_rel, dialogue_key_target, max_len, max_utter, max_target_size, max_mem_size, batch_size, is_test=False):
    padded_utters_id_w2v = None
    padded_utters_id_kb = None
    padded_target =[]
    
    padded_utters_id_w2v = np.asarray([dialogue_i for dialogue_i in dialogue_dict_w2v])
    padded_utters_id_kb = np.asarray([dialogue_i for dialogue_i in dialogue_dict_kb])
    
    if not is_test:
        padded_target = np.asarray([xi for xi in dialogue_target])
    else:
        padded_target = dialogue_target
    
    padded_sources = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_sources], dtype=np.int32)
    padded_rel = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_rel],  dtype=np.int32)
    padded_key_target = np.asarray([xi[:-1-max(0, len(xi)-max_mem_size)]+[kb_pad_idx]*(max(0,max_mem_size-len(xi)))+[xi[-1]] for xi in dialogue_key_target],  dtype=np.int32)
    
    return padded_utters_id_w2v, padded_utters_id_kb, padded_target, padded_sources, padded_rel, padded_key_target


def get_padded_seq_lens(padded_enc_w2v):
    '''Calculate the sequence len assuming padding (e.g. seq. in the dataset are already padded)'''
    
    enc_w2v_lens = np.asarray([(dialogue_i != pad_symbol_index).sum(axis=0) for dialogue_i in padded_enc_w2v])
    
    return enc_w2v_lens

'''
def get_mask(response):
    #get a binary mask, 0 for padding 1 o/w 
    
    response_mask = (response != pad_symbol_index).astype(int)
    
    return response_mask


def get_weights(batch_size, max_len, actual_len):
    remaining_len = max_len - actual_len
    weights = [[1.]*actual_len_i+[0.]*remaining_len_i for actual_len_i,remaining_len_i in zip(actual_len,remaining_len)]
    weights = np.asarray(weights)
    return weights

'''

def get_memory_weights(batch_size, max_mem_size, sources, rel, target):
    weights = np.ones((batch_size, max_mem_size))
    weights[np.where(sources==kb_pad_idx)] = 0.
    weights[np.where(rel==kb_pad_idx)] = 0.
    weights[np.where(target==kb_pad_idx)] = 0.
    weights[np.where(sources==nkb)] = 0.
    weights[np.where(rel==nkb)] = 0.
    weights[np.where(target==nkb)] = 0.
    
    return weights


def get_batch_data(data_dict, max_len=20, max_utter=1, max_target_size=10, batch_size=64, max_mem_size=2000, overriding_memory=None, is_test=False):
    
    data_dict = np.asarray(data_dict)
    
    batch_enc_w2v = data_dict[:,0]
    batch_enc_kb = data_dict[:,1]
    batch_target = data_dict[:,2]
    batch_orig_response = data_dict[:,3]
     
    #for the kvnn
    batch_sources = [x.split('|') for x in data_dict[:,4]]
    batch_rel = [x.split('|') for x in data_dict[:,5]]
    batch_key_target = [x.split('|') for x in data_dict[:,6]]
    
    if len(data_dict) % batch_size:
        # pad the batch
        batch_enc_w2v, batch_enc_kb, batch_target, batch_orig_response, batch_sources, batch_rel, batch_key_target = check_padding(batch_enc_w2v, batch_enc_kb, batch_target, batch_orig_response, batch_sources, batch_rel, batch_key_target, max_len, max_utter, max_mem_size, max_target_size, batch_size, is_test)

    padded_enc_w2v, padded_enc_kb, padded_target, padded_batch_sources, padded_batch_rel, padded_batch_key_target = get_utter_seq_len(batch_enc_w2v, batch_enc_kb, batch_target, batch_sources, batch_rel, batch_key_target, max_len, max_utter, max_target_size, max_mem_size, batch_size, is_test)
    
    #padded_weights = get_weights(batch_size, max_len, padded_response_length)
    
    #NOTE: not tranposed
    padded_memory_weights = get_memory_weights(batch_size, max_mem_size, padded_batch_sources, padded_batch_rel, padded_batch_key_target)
    
    padded_enc_w2v, padded_enc_kb, padded_target, padded_orig_target, padded_batch_sources, padded_batch_rel, padded_batch_key_target = transpose_utterances(padded_enc_w2v, padded_enc_kb, padded_target, padded_batch_sources, padded_batch_rel, padded_batch_key_target, max_mem_size, batch_size, is_test)
    
    enc_w2v_lens = get_padded_seq_lens(padded_enc_w2v)
    
    #padded_response_mask = get_mask(padded_response)
    
    return padded_enc_w2v, enc_w2v_lens, padded_enc_kb, padded_target, padded_orig_target, batch_orig_response, padded_memory_weights, padded_batch_sources, padded_batch_rel, padded_batch_key_target


def transpose_utterances(padded_enc_w2v, padded_enc_kb, padded_target, batch_sources, batch_rel, batch_key_target, max_mem_size, batch_size, is_test):

    batch_key_target = np.asarray(batch_key_target) # batch_size * max_mem_size
    # padded_target : batch_size * max_target_size
    
    '''
    # this still produces multitargets- double check
    if not is_test:
            #TODO:DOUBLE CHECK
            mapped_padded_target = np.zeros(batch_key_target.shape)
            #mapped_padded_target = np.full(batch_key_target.shape[0], max_mem_size-1)
            #mapped_padded_target = []
            for i in range(padded_target.shape[0]):
                #is_oov = True
                #multi_targets = []
                for j in range(padded_target.shape[1]):
                    if padded_target[i,j] in batch_key_target[i,:] and padded_target[i,j] != kb_pad_idx:
                        key_target_i = int(np.nonzero(batch_key_target[i,:] == padded_target[i,j])[0][0])
                        #mapped_padded_target[i] = key_target_i
                    else:
                        key_target_i = max_mem_size-1
                        
                    mapped_padded_target[i,j] = key_target_i
    '''
    
    if not is_test:
        #TODO:DOUBLE CHECK
        #mapped_padded_target = np.zeros(batch_key_target.shape)
        mapped_padded_target = np.full(batch_key_target.shape[0], max_mem_size-1)
        for i in range(padded_target.shape[0]):
            #is_oov = True
            for j in range(padded_target.shape[1]):
                if padded_target[i,j] in batch_key_target[i,:] and padded_target[i,j] != kb_pad_idx:
                    key_target_i = int(np.nonzero(batch_key_target[i,:] == padded_target[i,j])[0][0])
                    mapped_padded_target[i] = key_target_i
        #output will be like this: 
        '''
        tensor([1, 0, 1, 1, 9, 9, 9, 1, 0, 0, 9, 0, 1, 0, 2, 1, 9, 9, 0, 1, 1, 1, 1, 0,
        0, 9, 1, 1, 0, 1, 1, 9, 9, 4, 1, 1, 2, 0, 0, 9, 0, 1, 9, 1, 0, 0, 1, 0,
        1, 1, 9, 1, 0, 1, 1, 1, 9, 1, 0, 3, 1, 1, 1, 0])
        '''
        
       
    '''
    #use for multilabel classification
    if not is_test:
        mapped_padded_target = np.zeros(batch_key_target.shape)
        for i in range(padded_target.shape[0]):
            for j in range(padded_target.shape[1]):
                is_key_target_oov = True
                if padded_target[i,j] in batch_key_target[i,:] and padded_target[i,j] != kb_pad_idx:
                    key_target_i = int(np.nonzero(batch_key_target[i,:] == padded_target[i,j])[0][0])
                    mapped_padded_target[i,key_target_i] = 1 
                    is_key_target_oov = False
                if is_key_target_oov and padded_target[i,j] != kb_pad_idx: #not found key_target
                    #last mem entry is for oov
                    mapped_padded_target[i, max_mem_size-1] = 1
       
    '''
    
    padded_transposed_enc_w2v = padded_enc_w2v.transpose((1,2,0))
    padded_transposed_enc_kb = padded_enc_kb.transpose((1,2,0))
    
    #TODO: CHAGE NAME, IS NOT TRANSPOSED
    padded_transposed_target = mapped_padded_target
        
    if not is_test:
        padded_transposed_orig_target = padded_target.transpose((1,0))
    else:
        padded_transposed_orig_target = padded_target
        
    padded_batch_sources = np.asarray(batch_sources).transpose((1,0))
    padded_batch_rel = np.asarray(batch_rel).transpose((1,0))
    padded_batch_key_target = np.asarray(batch_key_target).transpose((1,0))

    return padded_transposed_enc_w2v, padded_transposed_enc_kb, padded_transposed_target, padded_transposed_orig_target, padded_batch_sources, padded_batch_rel, padded_batch_key_target


def batch_padding_context(data_mat, max_len, max_utter, pad_size):
    empty_data = [start_symbol_index, end_symbol_index]+[pad_symbol_index]*(max_len-2)
    empty_data = [empty_data]*max_utter
    empty_data_mat = [empty_data]*pad_size
    data_mat=data_mat.tolist()
    data_mat.extend(empty_data_mat)
    return data_mat


def batch_padding_target(data_mat, max_target_size, pad_size, is_test=False):
    if not is_test:
        empty_data = [kb_pad_idx] * max_target_size
        empty_data = [empty_data] * pad_size
        data_mat=data_mat.tolist()
        data_mat.extend(empty_data)
    else:
        if isinstance(data_mat, list):
            data_mat.extend(['']*pad_size)
        else:
            data_mat=data_mat.tolist()
            data_mat.extend(['']*pad_size)
            
    return data_mat


def batch_padding_orig_response(data_mat, pad_size):
    data_mat = data_mat.tolist()
    data_mat.extend(['']*pad_size)
    return data_mat


def batch_padding_memory_ent(data_mat, max_mem_size, pad_size):
    empty_data = [kb_pad_idx]*(max_mem_size)
    empty_data = [empty_data]*pad_size
    if not isinstance(data_mat, list):
        data_mat=data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat


def batch_padding_memory_rel(data_mat, max_mem_size, pad_size):
    empty_data = [kb_pad_idx]*(max_mem_size)
    empty_data = [empty_data]*pad_size
    if not isinstance(data_mat, list):
        data_mat=data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat


def check_padding(batch_enc_w2v, batch_enc_kb, batch_target, batch_orig_response, batch_sources, batch_rel, batch_key_target, max_len, max_utter, max_mem_size, max_target_size, batch_size, is_test=False):
    """completes the last batch by padding with empty data"""
    
    pad_size = batch_size - len(batch_target) % batch_size
    batch_enc_w2v = batch_padding_context(batch_enc_w2v, max_len, max_utter, pad_size)
    batch_enc_kb = batch_padding_context(batch_enc_kb, max_len, max_utter, pad_size)
    batch_target = batch_padding_target(batch_target, max_target_size, pad_size, is_test)
    batch_orig_response = batch_padding_orig_response(batch_orig_response, pad_size)
    # a dummy entry for OOM entities is added
    batch_sources = batch_padding_memory_ent(batch_sources, max_mem_size, pad_size) 
    batch_rel = batch_padding_memory_rel(batch_rel, max_mem_size, pad_size)
    batch_key_target = batch_padding_memory_ent(batch_key_target, max_mem_size, pad_size)
    
    return batch_enc_w2v, batch_enc_kb, batch_target, batch_orig_response, batch_sources, batch_rel, batch_key_target
