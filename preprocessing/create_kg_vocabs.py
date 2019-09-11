import os
import sys
from io import open
import pickle as pkl
import plac

#configure paths here
'''
config = {}
config['vocab_output_dir'] = "vocabs"
config['transe_dir'] = "datasets/transe_dir"
config['dataset_dir'] = 'my_datasets/preprocessed_data_simple_cqa/'
'''

PAD_KB_SYMBOL_INDEX = 0
PAD_KB_SYMBOL = '<pad_kb>'

NKB_SYMBOL_INDEX = 1
NKB_SYMBOL = '<nkb>'

def to_pkl(obj, file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        

'''
def process(root_path, ent_or_rel_file, ent_or_rel_id_map):
    
    PQ = {}
    freq = {}
    oov_freq = {}
    #i = 0
    
    dir_content = [os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    for d in dir_content:
        for root, dirs, files in os.walk(d):
            for name in dirs:
            
                ent_or_rel_path = os.path.join(root, name, name+ent_or_rel_file)
                ents_or_rels = open(ent_or_rel_path, "r", encoding='utf-8').read().strip().split('\n')
                
                for ent_or_rel in ents_or_rels:
                    parts = ent_or_rel.split("|")
                    for part in parts:
                        if part not in ent_or_rel_id_map:
                            if part not in oov_freq:
                                oov_freq[part] = 1
                            else:
                                oov_freq[part] += 1
                        
                        if part not in PQ:
                            #freq[part] = 1
                            PQ[part] = 1
                            #i += 1
                        else:
                            PQ[part] += 1
                            #freq[part] += 1
            
    return PQ, oov_freq
'''


def create_vocabs(dataset_dir, files_map, id_map):
    
    iv_and_oov = {}
    oov = {}
    
    split_dirs = os.listdir(dataset_dir)
    
    for split_dir in split_dirs:
        q_dirs = os.listdir(os.path.join(dataset_dir, split_dir))
    
        for q_dir in q_dirs: # scan every q dir.
            for data_file in files_map:
                file_path = os.path.join(dataset_dir, split_dir, q_dir, q_dir + data_file)
    
                ents_or_rels = open(file_path, "r", encoding="utf-8").read().strip().split("\n")
                
                for ent_or_rel in ents_or_rels:
                    parts = ent_or_rel.split("|")
                    for part in parts:
                        if part not in id_map:
                            if part not in oov:
                                oov[part] = 1
                            else:
                                oov[part] += 1
                            
                        # collect all (iv and oov)
                        if part not in iv_and_oov:
                            iv_and_oov[part] = 1
                        else:
                            iv_and_oov[part] += 1
    
    return iv_and_oov, oov
                

def get_transe(transe_dir, map_type):

    map_type_path = os.path.join(transe_dir, map_type)
    id_ent_or_rel_map = {PAD_KB_SYMBOL_INDEX:PAD_KB_SYMBOL, NKB_SYMBOL_INDEX: NKB_SYMBOL}
    id_ent_or_rel_map.update({(k+2):v for k, v in pkl.load(open(map_type_path, 'rb')).items()})
    ent_or_rel_id_map = {v: k for k, v in id_ent_or_rel_map.items()}

    return ent_or_rel_id_map
  
'''
@plac.annotations(
    vocab_type=('Vocabulary type', 'positional', None, str),
    output_file=('Output file name', 'positional', None, str)
)
'''
@plac.annotations(
    dataset_dir=('Directory of the data set', 'positional', None, str),
    transe_dir=('TransE directory', 'positional', None, str),
    output_dir=('Output directory to save the vocabs.', 'positional', None, str)
)
def main(dataset_dir, transe_dir, output_dir):
#def main(vocab_type, output_file):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # context and response files can contain subject or relation ents.
    files_types_map = {
        'ents': ['_context_entities.txt', '_response_entities.txt'], 
        'rels': ['_context_relations.txt']
    }
    
    type_ent_or_rel_map = {
        'ents': 'id_ent_map.pickle',   
        'rels': 'id_rel_map.pickle'
    }
  
    for type_map, files_map in files_types_map.items():
        id_map = get_transe(transe_dir, type_ent_or_rel_map[type_map])
    
        iv_and_oov, oov = create_vocabs(dataset_dir, files_map, id_map)
        
        # write vocabs to file
        output_file = os.path.join(output_dir, '%s_vocab.pkl' % type_map)
        output_file_oov = os.path.join(output_dir, '%s_oov.pkl' % type_map)
        
        to_pkl(iv_and_oov, output_file)
        to_pkl(oov, output_file_oov)
        
        
        
    
    '''
    # entities s U o , subjects and objects can be in the response
    type_map = {
        's': '_context_entities.txt',
        'r': '_context_relations.txt',
        'o': '_response_entities.txt'
    }
    
    if vocab_type not in type_map:
        sys.exit('type of vocab. must be s,r or o.')
    
    if not os.path.exists(config['vocab_output_dir']):
        os.mkdir(config['vocab_output_dir'])
    
    type_ent_or_rel_map = {
        's': 'id_ent_map.pickle',   
        'r': 'id_rel_map.pickle',
        'o': 'id_ent_map.pickle'
    }
    
    if not os.path.exists(config['transe_dir']):
        sys.exit('Dataset path do not exists.')
    
    id_ent_or_rel_map = {PAD_KB_SYMBOL_INDEX:PAD_KB_SYMBOL, NKB_SYMBOL_INDEX: NKB_SYMBOL}
    id_ent_or_rel_map.update({(k+2):v for k, v in pkl.load(open(os.path.join(config['transe_dir'], type_ent_or_rel_map[vocab_type]), 'rb')).items()})
    ent_or_rel_id_map = {v: k for k, v in id_ent_or_rel_map.items()}
    
    QP, oov_freq = process(config['dataset_dir'], type_map[vocab_type], ent_or_rel_id_map)
    
    out_file = os.path.join(config['vocab_output_dir'], '%s.pkl'%output_file)
    #out_file_freq = os.path.join(VOCAB_OUTPUT_DIR, 'freq_%s.pkl'%output_file)
    out_file_oov = os.path.join(config['vocab_output_dir'], 'oov_%s.pkl'%output_file)
    
    to_pkl(QP, out_file)
    #to_pkl(freq, out_file_freq)
    to_pkl(oov_freq, out_file_oov)
    '''
    
    

if __name__ == "__main__":
    plac.call(main)

    
    