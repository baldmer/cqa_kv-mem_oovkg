# TODO: restructure, this structure was the most convenient way to experiment in colab.
from __future__ import division
import operator
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import gensim
import random
import json
import plac

from read_data_multi import *

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

KB_PAD_IDX = 0
NKB = 1

class KVMemory(nn.Module):
    def __init__(self, config, pretrain_embed, mode=None):
        super(KVMemory, self).__init__()
        
        self.config = config
        
        self.dropout = nn.Dropout(p=config["dropout"])   
        
        self.C = nn.Linear(config['cell_size'], config['wikidata_embed_size']*2, bias=False)
        '''
        # same result as using a linear layer without bias

        self.C = torch.empty(config['cell_size'], config['wikidata_embed_size'] * 2, device=device, dtype=torch.float, requires_grad=True)
        if mode == "train":
            nn.init.xavier_normal_(self.C)
        '''
        
        self.R_1 = nn.Linear(config["wikidata_embed_size"], config["cell_size"], bias=False)
        self.B = nn.Linear(config['cell_size'], config['wikidata_embed_size'], bias=False)
        #encode the input query
        self.embed_A = nn.Embedding(config["input_size"], config["hidden_size"])
        
        if pretrain_embed is not None:
          #load pretrain embeddings
          print ('loading pre-trained word embeddings')
          self.embed_A.weight.data.copy_(torch.from_numpy(pretrain_embed))
          self.embed_A.weight.requires_grad = False
        
        #with transe
        self.gru = nn.GRU(config['hidden_size']+config["wikidata_embed_size"], config["cell_size"]) # output batch_size*cell_size
        #without transe
        #self.gru = nn.GRU(config['hidden_size'], config["cell_size"]) # output batch_size*cell_size
        
    def forward(self, enc_w2v, w2v_lens, enc_kb_emb, key_emb, key_target_emb, mem_weights):
            '''
            (9,274,339, 100) - ent_embed.pkl.npy
            (569, 100) - rel_embed.pkl.npy
            (3,000,000, 300) - w2v
            
            '''
            
            embed_q = self.embed_A(enc_w2v) #seq_len*bs*w2v_emb
            embed_q = torch.cat((embed_q, enc_kb_emb), 2) #seq_len*bs*(w2v_emb + wiki_emb)
            
            packed_q = nn.utils.rnn.pack_padded_sequence(embed_q, w2v_lens, enforce_sorted=False)
            #  pass through GRU
            _, q_state = self.gru(packed_q) #bs*cell_size
            q_state = self.dropout(q_state)
            q_state = q_state.squeeze() # from the encoder [1, hid_s, cell_s]
            
            #TODO: one hop is enough for our experiments
            for hop in range(self.config["hops"]):
                
                # --memory addressing--
                
                q_last = self.C(q_state) # batch_size * (2*wikidata_embed_size)
                #q_last = q_state.mm(self.C).clamp(min=0) # batch_size * (2*wikidata_embed_size)
                q_temp1 = q_last.unsqueeze(1) # batch_size * 1 * (2*wikidata_embed_size)
                
                #q_temp1 = q_temp1/q_temp1.norm(dim=2)[:,:,None] # bs*1*wiki*2  #L2 normalized
                #q_temp1[q_temp1 != q_temp1] = 0
                
                #key_emb #batch_size * size_memory * (2*wikidata_embed_size)
                
                #key_emb = key_emb/key_emb.norm(dim=2)[:,:,None] #bs*sm*wiki*2
                #key_emb[key_emb != key_emb] = 0
                
                #prod = key_emb * q_temp1
                #dotted_1 = torch.sum(prod, 2) #bs * ms
                #same as
                dotted = torch.bmm(q_temp1, key_emb.transpose(2,1)) #bs*1*ms
                dotted = dotted.squeeze(1) #bs*ms
                
                probs = F.softmax(dotted, dim=1) * mem_weights # bs * ms
                probs = torch.unsqueeze(probs, 1) # bs * 1 * ms
                
                # --value reading--
                
                #key_target_emb #bs * ms * wikidata_embed_size
                #values_emb = key_target_emb.transpose(2,1) #bs * wikidata_embed_size * ms, needs this shape when values_emb * probs
                #TODO: confirm, should be a weighted sum over value entries (e.g. dim 1), not of embedding dimension.
                #o_k = torch.sum(values_emb * probs, 2) #bs * wikidata_embed_size
                o_k = torch.bmm(probs, key_target_emb) #bs * 1 * wiki_size
                o_k = o_k.squeeze(1)
               
                #o_k = o_k.mm(self.R_1).clamp(min=0) #bs * cell_size
                o_k = self.R_1(o_k) #bs * cell_size
                
                q_state = torch.add(q_state, o_k)
                
                
            # find candidates, candidates are the value cells. (there is no other candidates in the data)
            
            #temp_1 = q_state.mm(self.B).clamp(min=0) #bs * wiki_embed
            temp_1 = self.B(q_state) #bs * wiki_embed
            temp_1 = temp_1.unsqueeze(1) # bs * 1 * wiki_embed
            
            #key_target_emb #bs * ms * wikidata_embed_size
            
            prob_mem = torch.sum(temp_1 * key_target_emb, 2) # batch_size * size_memory
            #prob_mem = F.log_softmax(prob_mem, dim=1) #NOTE: do not pass trough softmax if sigmoid is used
            #prob_mem = F.softmax(prob_mem, dim=1)
            
            mem_output = torch.sigmoid(prob_mem) * mem_weights
            
            return mem_output
            
    
def train(model, data, model_optimizer, loss_f, valid_data, config):
    
    model.train()
    
    print("reading transe embed.")
    
    ent_embed = np.load(os.path.join(config['transe_dir'], 'ent_embed.pkl.npy'))
    rel_embed = np.load(os.path.join(config['transe_dir'], 'rel_embed.pkl.npy'))

    new_row = np.zeros((1, config["wikidata_embed_size"]), dtype=np.float32)
    new_row_nkb = np.zeros((1, config["wikidata_embed_size"]), dtype=np.float32)
    #new_row_nkb = np.full((1, config['wikidata_embed_size']), 0.01, dtype=np.float32)
    
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <pad_kb>
    ent_embed = np.vstack([new_row_nkb, ent_embed]) # corr. to <nkb>

    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <pad_kb>
    rel_embed = np.vstack([new_row_nkb, rel_embed]) # corr. to <nkb>
    
    # here we use the OOV embeddings created offline
    if config['oov_ent_handler']:
        oov_ent_embed = np.load(os.path.join(config['transe_dir'], config['oov_ent_handler']))
        ent_embed = np.concatenate((ent_embed, oov_ent_embed))
    
    n_batches = int(math.ceil(len(data)/config['batch_size']))
  
    print_loss_total = 0
    best_valid_loss = float('inf')
    
    print ("Train started...")
    print ("Total batches %d, train size: %d" % (n_batches, len(data)))
    if not os.path.exists(config['save_path']):
    	os.mkdir(config['save_path'])    
    out_file_loss_valid = open(os.path.join(config['save_path'], config['save_name_prefix']+'_train_valid_loss.txt'), "w")
  
    for epoch in range(config["max_epochs"]):
        
        if (epoch + 1) % 10 == 0:
            adjust_learning_rate(model_optimizer, epoch, config["lr"])
        
        for i_batch in range(n_batches):
              
            model_optimizer.zero_grad()
            
            batch_raw = data[i_batch*config['batch_size']:(i_batch+1)*config['batch_size']]
            batch = get_batch_data(batch_raw, max_mem_size=config["max_mem_size"], batch_size=config["batch_size"])
        
            enc_w2v, w2v_lens, enc_kb, target, target_lens, orig_target, orig_response, mem_weights, sources, rel, key_target = batch
                  
            # convert to torch tensors
            enc_w2v = torch.LongTensor(enc_w2v)
            w2v_lens = torch.Tensor(w2v_lens)
            enc_kb = torch.LongTensor(enc_kb)
            mem_weights = torch.FloatTensor(mem_weights)
            
            target = torch.FloatTensor(target) # BCEWithLogitsLoss requires float
            #target = torch.LongTensor(target)
            #response = torch.LongTensor(response)
            
            # send to device
            enc_w2v = enc_w2v.to(device)
            w2v_lens = w2v_lens.to(device)
            enc_kb = enc_kb.to(device)
            target = target.to(device)
            mem_weights = mem_weights.to(device)
            #response =  response.to(device)
            
            # get the emb of all the (subj, rel, obj) in the batch
            ent_emb = torch.FloatTensor([np.array([ent_embed[i] for i in ent_i]) for ent_i in sources]) # size_memory * batch_size * wikidata_embed_size
            ent_emb = ent_emb.transpose(1, 0)
            rel_emb = torch.FloatTensor([np.array([rel_embed[i] for i in rel_i]) for rel_i in rel])
            rel_emb = rel_emb.transpose(1, 0)
            
            key_emb = torch.cat((ent_emb, rel_emb), 2) # batch_size * size_memory * (2*wikidata_embed_size)
            #memory_size * batch_size * wikidata_embed_size
            key_target_emb = torch.FloatTensor([np.array([ent_embed[i] for i in key_target_i]) for key_target_i in key_target]) 
            key_target_emb = key_target_emb.transpose(1,0) #batch_size * size_memory * wikidata_embed_size
            
            enc_kb_emb = torch.FloatTensor([np.array([ent_embed[i] for i in enc_kb_i]) for enc_kb_i in enc_kb[0]]) # seq_len*batch_size*wikidata_emb 
            #to device
            key_emb = key_emb.to(device)
            key_target_emb = key_target_emb.to(device)
            enc_kb_emb = enc_kb_emb.to(device)
            
            #enc_w2v[1] max_seq_len * batch_size, use only one question as input
            probs = model.forward(enc_w2v[0], w2v_lens[0], enc_kb_emb, key_emb, key_target_emb, mem_weights)
            curr_loss  = loss_f(probs, target) #summed loss
            avg_batch_loss = curr_loss.item() / config['batch_size']
            curr_loss.backward()
            
            _ = nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad'])
            
            model_optimizer.step()
            
            if i_batch % config['print_every'] == 0 and i_batch > 0:
                print ("Epoch %d, batch %d,  Avg. train loss(over batch) = %.3f" % (epoch, i_batch, avg_batch_loss))
                
        # validate every configured epoch, save the best validation loss

        if epoch % config['valid_every_epoch'] == 0:
            with torch.no_grad():
                
                print ("Validating...")
                
                model.eval()
                valid_loss = 0
                random.shuffle(valid_data)
                n_valid_batches = int(math.ceil(len(valid_data)/config['batch_size']))
                
                print ("Total batches %d, valid size: %d" % (n_valid_batches, len(valid_data)))
                
                for j_batch in range(n_valid_batches):
                #for j_batch in range(1): # to evaluate subjectively only
                    valid_batch_raw = valid_data[j_batch*config['batch_size']:(j_batch+1)*config['batch_size']]
                    valid_batch = get_batch_data(valid_batch_raw, max_mem_size=config["max_mem_size"], batch_size=config["batch_size"])
                
                    enc_w2v, w2v_lens, enc_kb, target, target_lens, orig_target, orig_response, mem_weights, sources, rel, key_target = valid_batch
                
                    # convert to torch tensors
                    enc_w2v = torch.LongTensor(enc_w2v)
                    w2v_lens = torch.Tensor(w2v_lens)
                    #enc_kb = torch.LongTensor(enc_kb)
            
                    target = torch.FloatTensor(target) # BCEWithLogitsLoss requires float
                    #target = torch.LongTensor(target)
                    #response = torch.LongTensor(response)
                    mem_weights = torch.FloatTensor(mem_weights)
            
                    # send to device
                    enc_w2v = enc_w2v.to(device)
                    w2v_lens = w2v_lens.to(device)
                    #enc_kb = enc_kb.to(device)
                    target = target.to(device)
                    #response =  response.to(device)
                    mem_weights = mem_weights.to(device)
            
                    # get the emb of all the (subj, rel, obj) in the batch
                    ent_emb = torch.FloatTensor([np.array([ent_embed[i] for i in ent_i]) for ent_i in sources]) # size_memory * batch_size * wikidata_embed_size
                    ent_emb = ent_emb.transpose(1, 0)
                    rel_emb = torch.FloatTensor([np.array([rel_embed[i] for i in rel_i]) for rel_i in rel])
                    rel_emb = rel_emb.transpose(1, 0)
            
                    key_emb = torch.cat((ent_emb, rel_emb), 2) # batch_size * size_memory * (2*wikidata_embed_size)
                    key_target_emb = torch.FloatTensor([np.array([ent_embed[i] for i in key_target_i]) for key_target_i in key_target])
                    key_target_emb = key_target_emb.transpose(1,0) #batch_size * size_memory * wikidata_embed_size
                
                    enc_kb_emb = torch.FloatTensor([np.array([ent_embed[i] for i in enc_kb_i]) for enc_kb_i in enc_kb[0]]) # seq_len*batch_size*wikidata_emb
                
                    #to device
                    key_emb = key_emb.to(device)
                    key_target_emb = key_target_emb.to(device)
                    enc_kb_emb = enc_kb_emb.to(device)
            
                    #enc_w2v[0] max_seq_len * batch_size, use only one question as input
                    #TODO question has a <kb> placeholder, that will be considered as oov
                    probs = model.forward(enc_w2v[0], w2v_lens[0], enc_kb_emb, key_emb, key_target_emb, mem_weights)
                    curr_valid_loss = loss_f(probs, target)
                    valid_loss += curr_valid_loss.item() # summed loss
                    
                overall_valid_loss = valid_loss/len(valid_data)
                print ("Overall valid loss = %.4f" % overall_valid_loss)
                
                if overall_valid_loss < best_valid_loss:
                    print ("New best validation loss found, saving model...")
                    
                    best_valid_loss = overall_valid_loss
                    # save the model
                    if not os.path.exists(config['save_path']):
                        os.mkdir(config['save_path'])
                        
                    torch.save(model.state_dict(), os.path.join(config['save_path'], '%s_checkpoint_%d.pt' % (config["save_name_prefix"], epoch)))
                
                # save stats to file, current train batch loss vs overall valid loss
                out_file_loss_valid.write("%s\t%s\n" % (avg_batch_loss, overall_valid_loss))
                
        model.train() # keep training
                
                
def adjust_learning_rate(optimizer, epoch, lr):
    lr = lr / (2 ** (epoch // 10))
    
    print ("Adjust lr to %s" % lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model, data, config):
  
    #wikidata_id_name_map = json.load(open('data/items_wikidata_n.json'))
  
    ent_embed = np.load(os.path.join(config['transe_dir'], 'ent_embed.pkl.npy'))
    rel_embed = np.load(os.path.join(config['transe_dir'], 'rel_embed.pkl.npy'))    
    
    new_row = np.zeros((1, config["wikidata_embed_size"]), dtype=np.float32)
    new_row_nkb = np.zeros((1, config["wikidata_embed_size"]), dtype=np.float32)
    #new_row_nkb = np.full((1, config['wikidata_embed_size']), 0.01, dtype=np.float32)
    
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <pad_kb>
    ent_embed = np.vstack([new_row_nkb, ent_embed]) # corr. to <nkb>

    rel_embed = np.vstack([new_row, rel_embed]) # corr. to <pad_kb>
    rel_embed = np.vstack([new_row_nkb, rel_embed]) # corr. to <nkb>
    
    # here we use the OOV embeddings created offline
    if config['oov_ent_handler']:
        oov_ent_embed = np.load(os.path.join(config['transe_dir'], config['oov_ent_handler']))
        ent_embed = np.concatenate((ent_embed, oov_ent_embed))
    
    n_batches = int(math.ceil(len(data)/config['batch_size']))
    
    #csv_w = csv.writer(open("models/test_output.csv","w", newline=''))
    #csv_w.writerow("target_index, pred_index, target_ent, pred_ent, orig_response")
    
    #out_file_path = os.path.join(config['save_path'], config['out_test_file'])
    if not os.path.exists(config['metrics_dir']):
    	os.mkdir(config['metrics_dir'])
    metrics_path = os.path.join(config['metrics_dir'], config['out_test_file'])
    out_file = open(metrics_path, "w")
    #out_file.write("target_idx\tpred_idx\ttarget_ent_id\tpred_ent_id\tall_target_ent\torig_resp\n")
                    
    for i_batch in range(n_batches):
        batch_raw = data[i_batch*config['batch_size']:(i_batch+1)*config['batch_size']]
        test_batch = get_batch_data(batch_raw, max_mem_size=config["max_mem_size"])
                
        enc_w2v, w2v_lens, enc_kb, target, target_lens, orig_target, orig_response, mem_weights, sources, rel, key_target = test_batch
                    
        # convert to torch tensors
        enc_w2v = torch.LongTensor(enc_w2v)
        w2v_lens = torch.Tensor(w2v_lens)
        #enc_kb = torch.LongTensor(enc_kb)
        mem_weights = torch.FloatTensor(mem_weights)

        #target = torch.FloatTensor(target) # BCEWithLogitsLoss requires float
        #target = torch.LongTensor(target)
        #response = torch.LongTensor(response)

        # send to device
        enc_w2v = enc_w2v.to(device)
        w2v_lens = w2v_lens.to(device)
        #enc_kb = enc_kb.to(device)
        #target = target.to(device)
        #response =  response.to(device)
        mem_weights = mem_weights.to(device)

        # get the emb of all the (subj, rel, obj) in the batch
        ent_emb = torch.FloatTensor([np.array([ent_embed[i] for i in ent_i]) for ent_i in sources]) # size_memory * batch_size * wikidata_embed_size
        ent_emb = ent_emb.transpose(1, 0)
        rel_emb = torch.FloatTensor([np.array([rel_embed[i] for i in rel_i]) for rel_i in rel])
        rel_emb = rel_emb.transpose(1, 0)

        key_emb = torch.cat((ent_emb, rel_emb), 2) # batch_size * size_memory * (2*wikidata_embed_size)
        key_target_emb = torch.FloatTensor([np.array([ent_embed[i] for i in key_target_i]) for key_target_i in key_target])
        key_target_emb = key_target_emb.transpose(1,0) #batch_size * size_memory * wikidata_embed_size

        enc_kb_emb = torch.FloatTensor([np.array([ent_embed[i] for i in enc_kb_i]) for enc_kb_i in enc_kb[0]]) # seq_len*batch_size*wikidata_emb

        #to device
        key_emb = key_emb.to(device)
        key_target_emb = key_target_emb.to(device)
        enc_kb_emb = enc_kb_emb.to(device)

        #enc_w2v[0] max_seq_len * batch_size, use only one question as input
        probs = model.forward(enc_w2v[0], w2v_lens[0], enc_kb_emb, key_emb, key_target_emb, mem_weights)

        #_, pred_indexes = torch.max(probs, 1)
        #np_pred_indexes = pred_indexes.cpu().detach().numpy()
        
        preds = []
        for i, len_i in enumerate(target_lens):
            _, pred_indexes = torch.topk(probs[i].unsqueeze(0), int(len_i), 1)
            preds.append(pred_indexes.squeeze(0).cpu().detach().numpy())
        
        #TODO: try is_test=True param
        orig_target = orig_target.T # batch_size * max_target_len
        
        # entities idexed in target can be found on key_target
        key_target = key_target.T # batch_size * memory_size
        gold_entities = []
        pred_entities = []
        
        # get value indexed by target_id in memory key_target_id
        for preds_i, key_target_i, orig_target_i, orig_resp_i in zip(np_pred_indexes, key_target, orig_target, orig_response):
            row = []
            row.append('%s' % '|'.join([str(ident) for ident in key_target_i[preds_i]]))
            row.append('%s' % '|'.join([str(ident) for ident in orig_target_i if ident not in [KB_PAD_IDX]])) #EXCLUDE OOV IN ORIG GOLD TARGET
            row.append('%s' % '|'.join([str(ident) for ident in key_target_i]))
            row.append(orig_resp_i)
            #csv_w.writerow(csv_row)
            out_file.write("\t".join(row))


def init_embeddings(vocab, embed_size, pretrain_embed):
    vocab_len = len(vocab.keys())
    #vocab_init.embed = np.zeros((vocab_len, embed_size))
    vocab_init_embed = np.empty([vocab_len, embed_size], dtype=np.float32)
    #word2vec_pretrain_embed = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    
    for i in range(vocab_len):
        if vocab[i] in pretrain_embed:
            vocab_init_embed[i,:] = pretrain_embed[vocab[i]]
        else:
            vocab_init_embed[i,:] = np.random.rand(1, embed_size).astype(np.float32)
        #vocab_init_embed[i,:] = np.zeros((1, embed_size)).astype(np.float32)
        
    return vocab_init_embed

#torch.autograd.set_detect_anomaly(True)

#load embeddings in text file
def load_text_embeddings(file_name):
  
    print("Loading Glove Model")
    
    f = open(file_name,'r')
    glove = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]], dtype='float32')
        glove[word] = embedding
        
    print("Done.", len(glove)," words loaded!")
    print('Done, %d words loaded.' % len(glove))
    
    return glove


vocab = pkl.load(open("vocabs/vocab.pkl", "rb"))

config = {
    'wikidata_embed_size': 100,
    'max_epochs': 10,
    'batch_size': 64,
    'lr': 0.0001, # 'lr': 0.0001 worked well for single class
    #'max_seq_len': 10,
    #'max_utter': 1,
    #'max_target_size':10,
    'max_mem_size': 10, # grater size might improve scores, take longer to train.
    'input_size': len(vocab.keys()),
    'hidden_size': 300, # must be same dim as pre-train
    'cell_size': 400, # must be hidden_size + wiki_embed_size 
    'hops': 1, # all of our experiments are with 1 hop
    'print_every': 500,
    'valid_every_epoch': 1, # save best validation loss model
    'save_path':'models',
    'out_test_file': '', # will be "model_name"+"out_test.txt"
    'clip_grad': 5,
    'dropout': 0.2,
    'pretrain_word_model': "word2vec",  # word2vec, glove
    'save_name_prefix': 'no_oov_new_mem',
    'train_data_file': "datasets/no_oov_handling_new_mem/train.pkl",
    'test_data_file': "datasets/no_oov_handling_new_mem/test.pkl",
    'valid_data_file': "datasets/no_oov_handling_new_mem/valid.pkl",
    #'oov_ent_handler': "oov_text_matching_ent_embed.npy", # specify embeddings file for oov, assumed to be in transe_dir
    'oov_ent_handler': None, 
    'transe_dir': "datasets/transe_dir",
    'embed_dir': "datasets/embed_dir",
    'metrics_dir': 'metrics'
}

@plac.annotations(
    mode=('Mode type', 'positional', None, str),
    model_file=('Model', "option", "mdl", str)
    #norm_type=("Normalization type", "option", "t", str)
)
def main(mode, model_file=None):
    
    # load pretrain embeddings
    pretrain_embed = None
    if config['pretrain_word_model'] == 'word2vec':
        # embed_file = "GoogleNews-vectors-negative100.txt"
        embed_file = 'GoogleNews-vectors-negative300.bin'
        embed_path = os.path.join(config['embed_dir'], embed_file)
        pretrain_embed = gensim.models.KeyedVectors.load_word2vec_format(embed_path, binary=True)
        #pretrain_embed = load_text_embeddings(embed_file)
    elif config['pretrain_word_model'] == 'glove':
        embed_file = 'GoogleNews-vectors-negative100.txt'
        embed_path = os.path.join(config['embed_dir'], embed_file)
        pretrain_embed = load_text_embeddings(embed_file) # TODO: use glove model
    
    if pretrain_embed:
        pretrain_embed = init_embeddings(vocab, config['hidden_size'], pretrain_embed)
    
    # create model
    model = KVMemory(config, pretrain_embed, mode)
    model = model.to(device)

    #model_optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    #optim.SGD(lr=1e-2, momentum=0.9, nesterov=True)
    # TODO: try # weight_decay = 0.0005
    model_optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    #model_optimizer.load_state_dict(config['save_optimizer'])

    #loss_f = nn.NLLLoss(reduction="sum") #default mean
    #loss_function = nn.BCEWithLogitsLoss(reduction="mean") #with no softmax gives decent output, loss is very low.
    loss_f = nn.BCELoss(reduction="sum")

    if mode == 'train':
        train_data = pkl.load(open(config['train_data_file'], "rb"))
        valid_data = pkl.load(open(config['valid_data_file'], "rb"))

        random.shuffle(train_data)
        #random.shuffle(valid_data)
        
        if model_file is not None:
            model.load_state_dict(torch.load(model_file))

        train(model, train_data, model_optimizer, loss_f, valid_data, config)
        
    elif mode == 'test':

        #load model and test it
        print ('Testing...')   
     
        if model_file is None:
            exit("Must provide the name of the model to test.")
        
        test_data = pkl.load(open(config['test_data_file'], "rb"))
        model.load_state_dict(torch.load(model_file))
        '''
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        print("Optimizer's state_dict:")
        for var_name in model_optimizer.state_dict():
            print(var_name, "\t", model_optimizer.state_dict()[var_name])
        '''
        name_model_file = os.path.basename(model_file)
        config['out_test_file'] = os.path.splitext(name_model_file)[0] + '_out_test.txt' 

        test(model, test_data, config)


if __name__ == "__main__":
    plac.call(main)
