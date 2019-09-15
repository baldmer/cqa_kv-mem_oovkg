import os
from io import open
#import nltk
import pickle as pkl
import json
import plac

try:
	import nltk
except:
	print ("WARNING: NLTK not available, split() will be used")

WIKIDATA_DIR = "datasets/wikidata_dir"
TRANSE_DIR = "datasets/transe_dir"
VOCAB_FILE = "vocabs/vocab.pkl"

PAD_KB_SYMBOL_INDEX = 0
PAD_KB_SYMBOL = '<pad_kb>'

NKB_SYMBOL_INDEX = 1
NKB_SYMBOL = '<nkb>'

KB_OV_IDX = 1
#kb_rel_ov_idx = 1

START_WORD_SYMBOL = '</s>'

END_WORD_SYMBOL = '</e>'
END_WORD_ID = 1

KB_WORD_ID = 4

PAD_SYMBOL = '<pad>'
PAD_WORD_ID = 3

UNK_WORD_ID = 2

MAX_LEN = 20
MAX_TARGET_SIZE = 10
MAX_MEM_SIZE = 1500
NUM_TRANSE_EMBED = 9274339

def pad_or_clip_utterance(utterance):
        
    if len(utterance)>(MAX_LEN-2):
        utterance = utterance[:(MAX_LEN-2)]
        utterance.append(END_WORD_SYMBOL)
        utterance.insert(0, START_WORD_SYMBOL)
    elif len(utterance)<(MAX_LEN-2):
        pad_length = MAX_LEN - 2 - len(utterance)
        utterance.append(END_WORD_SYMBOL)
        utterance.insert(0, START_WORD_SYMBOL)
        utterance = utterance+[PAD_SYMBOL]*pad_length
    else:
        utterance.append(END_WORD_SYMBOL)
        utterance.insert(0, START_WORD_SYMBOL)
    
    return utterance


def pad_or_clip_target(target_list):
    
    if len(target_list) > MAX_TARGET_SIZE:
        target_list = target_list[:MAX_TARGET_SIZE]
    elif len(target_list) < MAX_TARGET_SIZE:
        pad_length = MAX_TARGET_SIZE - len(target_list)
        target_list = target_list + [PAD_KB_SYMBOL] * pad_length
        
    return target_list
    

def pad_or_clip_memory(mem_list):
    if len(mem_list) > MAX_MEM_SIZE:
        mem_list = mem_list[:MAX_MEM_SIZE]
    mem_list = mem_list+[NKB_SYMBOL]
    
    return mem_list


def isQid(input_str, entity_id_map):
    if input_str.upper() not in entity_id_map:
        return False

    if len(input_str) == 0:
        return False

    if input_str in [PAD_KB_SYMBOL, NKB_SYMBOL]:
        return True

    char0 = input_str[0]
    rem_chars = input_str[1:]
    
    if char0 != 'Q' and char0 !='q':
        return False
    
    try:
        x = int(rem_chars)
    except:
        return False
        
    return True 


def to_pickle(data, file_name):
    
    with open(file_name, 'wb') as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)


def binarize_context(context, params):
    ''' binarize context utterance/include pids.'''
    
    wiki_qid_to_name = params['wiki_qid_to_name']
    vocab = params['vocab']
    entity_id_map = params['entity_id_map']
    bad_qids = params['bad_qids']
                        
    context = context.lower().strip()

    binarized_context = []
    binarized_context_kg = []
    num_unk_words = 0
    try:
        #import nltk
        utter_words = nltk.word_tokenize(context)
    except:
        utter_words = context.split(" ")

    utter_words = pad_or_clip_utterance(utter_words)

    if END_WORD_SYMBOL not in utter_words:
        print('utterance: %s' % (context))
        raise Exception('utterance does not have end symbol')

    utter_word_ids_1 = []
    utter_word_ids_2 = []

    for word in utter_words:
        if word in bad_qids:
            word = wiki_qid_to_name[word]
            
        if isQid(word, entity_id_map):
            utter_word_ids_1.append(KB_WORD_ID)
            utter_word_ids_2.append(entity_id_map[word.upper()])
        else:
            word_id = vocab.get(word, UNK_WORD_ID)
            #freqs[word_id] += 1
            utter_word_ids_1.append(word_id)
            
            if word_id != PAD_WORD_ID:
                utter_word_ids_2.append(KB_OV_IDX)
            else:
                utter_word_ids_2.append(entity_id_map[PAD_KB_SYMBOL])
            
        num_unk_words += 1 * (word_id == UNK_WORD_ID)

    if END_WORD_ID not in utter_word_ids_1:
        print('orig. utterance: %s with ids:' % (context))
        print(utter_word_ids_1)
        raise Exception('utterance word ids_1 does not have end word id')

    num_terms = len(utter_words)

    binarized_context.append(utter_word_ids_1)
    binarized_context_kg.append(utter_word_ids_2)
    
    return binarized_context, binarized_context_kg, num_unk_words, num_terms


def binarize_kg_target(target, entity_id_map):
    '''these are the ground truth entities'''
     
    target_ids = []
    target = target.rstrip()
    
    if len(target) > 0:
        target = target.split("|")
    else:
        target = []
        
    target = pad_or_clip_target(target)

    for qid in target:
        if isQid(qid, entity_id_map):
            ident = entity_id_map[qid]
            target_ids.append(ident)
        elif qid != PAD_KB_SYMBOL:
            target_ids.append(KB_OV_IDX)
        else:
            target_ids.append(entity_id_map[PAD_KB_SYMBOL])
        
    return target_ids
    

def binarize_source(source, entity_id_map):
    ''' OOV entities are already filtered from cand. generation'''
    
    source_word_ids = []
    source = source.rstrip()

    if len(source) > 0:
        source = source.split('|')
    else:
        source = []

    source_words = pad_or_clip_memory(source)

    for word in source_words:
        word_id = entity_id_map[word]
        source_word_ids.append(word_id)

    source_word_ids = "|".join([str(x) for x in source_word_ids])
    
    return source_word_ids


def binarize_relation(relation, rel_id_map):
    ''' OOV rels are already filtered'''
    
    relation_word_ids = []
    relation = relation.rstrip()
  
    if len(relation) > 0:
        relation = relation.split('|')
    else:
        relation = []
    
    relation_words = pad_or_clip_memory(relation)
    
    for word in relation_words:
        word_id = rel_id_map[word]
        relation_word_ids.append(word_id)
    
    relation_word_ids = "|".join([str(x) for x in relation_word_ids])
    
    return relation_word_ids


def binarize_key_target(key_target, entity_id_map):
    '''OOV entities are already filtered'''
    
    key_target_word_ids = []
    key_target = key_target.rstrip()

    if len(key_target) > 0:
        key_target = key_target.split('|')
    else:
        key_target = []

    key_target_words = pad_or_clip_memory(key_target)

    for word in key_target_words:
        word_id = entity_id_map[word]
        key_target_word_ids.append(word_id)

    key_target_word_ids = "|".join([str(x) for x in key_target_word_ids])
    
    return key_target_word_ids


@plac.annotations(
    input_corpus=('Input dataset split', 'positional', None, str),
    output=('Binarized output file', 'positional', None, str),
    oov_id_ent_map=('File containing oov embeds. (id_ent_map)', 'option', 'oov', str)
)
def main(input_corpus, output, oov_id_ent_map=None):
    
    if not os.path.exists(input_corpus):
        exit("Corpus path does not exists")
    
    binarized_corpus = []
    num_unk_words = 0
    num_terms = 0
    num_instances = 0
    # yes/no
    bad_qids = set(['Q184386','Q1541554','Q540955','Q2620241','Q742391'])
    bad_qids.update(pkl.load(open(os.path.join(WIKIDATA_DIR, 'wikidata_entities_with_digitnames.pkl'), 'rb')))

    wikidata_qid_to_name = json.load(open(os.path.join(WIKIDATA_DIR, 'items_wikidata_n.json')))
    
    vocab = pkl.load(open(VOCAB_FILE, 'rb'))
    vocab_w2id = {v:k for k,v in vocab.items()}
    
    id_entity_map = {PAD_KB_SYMBOL_INDEX:PAD_KB_SYMBOL, NKB_SYMBOL_INDEX: NKB_SYMBOL}
    id_entity_map.update({(k+2):v for k, v in pkl.load(open(os.path.join(TRANSE_DIR, 'id_ent_map.pickle'), 'rb')).items()})
	
    if oov_id_ent_map:
        try:
            id_entity_map.update({(k+2+NUM_TRANSE_EMBED):v for k,v in pkl.load(open(oov_id_ent_map,'rb')).items()})
        except:
            exit('Incorrect oov file name')    
    # NOTE: comment for no oov handling
    #id_entity_map.update({(k+2+NUM_TRANSE_EMBED):v for k,v in pkl.load(open(dir_name+'/v2_oov_id_ent_map.pickle','rb')).items()})
    
    entity_id_map = {v: k for k, v in id_entity_map.items()}
    
    id_rel_map = {PAD_KB_SYMBOL_INDEX:PAD_KB_SYMBOL, NKB_SYMBOL_INDEX: NKB_SYMBOL}
    id_rel_map.update({(k+2):v for k, v in pkl.load(open(os.path.join(TRANSE_DIR, 'id_rel_map.pickle'), 'rb')).items()})
    rel_id_map = {v: k for k, v in id_rel_map.items()}
    
    for root, dirs, files in os.walk(input_corpus):
        
        for dir_name in dirs:
            
            print ("Processing dir: %s" % dir_name)            
            context_path = os.path.join(root, dir_name, dir_name+'_context.txt')
            #context = open(context_path, encoding='utf-8').read().split('\n')
            response_entities_path = os.path.join(root, dir_name, dir_name+'_response_entities.txt')
            #response_entities = open(response_entities_path, encoding='utf-8').read().split('\n')
            orig_response_path = os.path.join(root, dir_name, dir_name+'_orig_response.txt')
            #orig_response = open(orig_response_path, encoding='utf-8').read().split('\n')
            sources_path = os.path.join(root, dir_name, dir_name+'_sources.txt')
            relations_path = os.path.join(root, dir_name, dir_name+'_relations.txt')
            key_targets_path = os.path.join(root, dir_name, dir_name+'_key_targets.txt')
    
            #NOTE: there is no response with vocab ids, not needed if KG response entities used.
            with open(context_path) as contextlines, open(response_entities_path) as targetlines, open(orig_response_path) as orig_responselines, open(sources_path) as source_lines, open(relations_path) as relation_lines, open(key_targets_path) as key_target_lines:
                
                for context, target, orig_response, source, relation, key_target in zip(contextlines, targetlines, orig_responselines, source_lines, relation_lines, key_target_lines):
                    
                    num_instances += 1
                    
                    # binarize context
                    
                    context_params = {
                        'wiki_qid_to_name': wikidata_qid_to_name,
                        'vocab' : vocab_w2id,
                        'entity_id_map': entity_id_map,
                        'bad_qids': bad_qids
                    }
                    
                    binarized_context, binarized_context_kg, num_unk_words_context, num_terms_context = binarize_context(context, context_params)
                    num_unk_words += num_unk_words_context
                    num_terms += num_terms_context
                    
                    # binarize ground truth.
                    target =  binarize_kg_target(target, entity_id_map)
                   
                    #TODO: use orig_response to get the ids of response and binarize.
                    
                    # binarize source/subject entities
                    source_word_ids = binarize_source(source, entity_id_map)
                    
                    # binarize relations
                    relation_word_ids = binarize_relation(relation, rel_id_map)
                    
                    # binarize key_target
                    key_target_word_ids = binarize_key_target(key_target, entity_id_map)
                    
                    binarized_corpus.append([binarized_context, binarized_context_kg, target, orig_response, source_word_ids, relation_word_ids, key_target_word_ids])
                    
        #create a binarized file per folder if needed
    
    to_pickle(binarized_corpus, output)
    
    print ("Number of instances: %d \n" % num_instances)
    print ("Number of words: %d \n" % num_terms)
    print ("Number of UNK words %d \n" % num_unk_words)
    

if __name__ == "__main__":
    plac.call(main)
