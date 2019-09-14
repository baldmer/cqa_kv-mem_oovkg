'''
Find oov entities embeddings in word embeddings.
Uses text matching approach.
'''
import os
import pickle as pkl
import json
import numpy as np
import gensim
import plac

pad_kb_symbol_index = 0
pad_kb_symbol = '<pad_kb>'
nkb_symbol_index = 1
nkb_symbol = '<nkb>'

#configure paths
config = {}
config["transe_dir"] = "datasets/transe_dir"
config["vocabs_dir"] = "vocabs"
config["wikidata_dir"] = "datasets/wikidata_dir"
config["embed_dir"] = "datasets/embed_dir"
config["topn_sim"] = 10


def load_transe(wiki_items):
    
    id_entity_map = {pad_kb_symbol_index:pad_kb_symbol, nkb_symbol_index: nkb_symbol}
    id_entity_map.update({(k+2):v for k,v in pkl.load(open(os.path.join(config['transe_dir'], 'id_ent_map.pickle'),'rb')).items()})
    #id_entity_map = pkl.load(open('data/id_ent_map.pickle','rb'))
    entity_id_map = {v: k for k, v in id_entity_map.items()}
    
    ent_embed = np.load(os.path.join(config['transe_dir'], 'ent_embed.pkl.npy'))
    new_row = np.zeros((1, 100), dtype=np.float32)
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <pad_kb>
    ent_embed = np.vstack([new_row, ent_embed]) # corr. to <nkb>
    
    label_entity_map = {wiki_items[qid]:qid for qid, _ in entity_id_map.items()}
    
    return entity_id_map, label_entity_map, ent_embed


def load_word_emebeds(algo):
    
    #TODO: suport other types of embedding models
   
    if algo == 'w2v':
        # read w2v into gensim
        w2v_path = os.path.join(config['embed_dir'], 'GoogleNews-vectors-negative300.bin.gz')
        embeds = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    else:
        exit("Embedding model not valid.")
        
    return embeds
    

def load_wiki_items():
    
    path = os.path.join(config["wikidata_dir"], "items_wikidata_n.json")
    
    with open(path) as f:
        items = json.load(f)
    
    return items


def load_oov_entities(wiki_items):
    '''return all oov entities maped to its label'''
    
    ent_oov_path = os.path.join(config['vocabs_dir'], 'entities_oov.pkl')
    ent_oov = pkl.load(open(ent_oov_path, 'rb'))
    
    qids_labels_map = {qid:wiki_items[qid] for qid, _ in ent_oov.items()}
    
    return qids_labels_map


def to_pkl(obj, file_name):
    with open(file_name, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    
def search_in(word_embeds, oov_entities):
    """search oov entities with word embeddings"""
    
    entity_label_found = {}
    
    for ent, label in oov_entities.items():
        label_cap = label.capitalize()
        if label in word_embeds:
            # single words
            entity_label_found[ent] = label
        elif label_cap in word_embeds:
            # first character in capital
            entity_label_found[ent] = label_cap
        else:
            # sentences (words_separated by _)
            sentence = '_'.join(label.split(' '))
            if sentence in word_embeds:
                entity_label_found[ent] = sentence
                
    return entity_label_found


@plac.annotations(
    out=('Output file name', 'positional', None, str),
    algo=('Word embedding algorithm type [w2v, glove, fasttext]', 'option', 'algo', str)
)
def main(out, algo='w2v'):
    
    print ("Loading wiki items")
    wiki_items = load_wiki_items()
    
    print ("Loading word embeddings")
    word_embeds = load_word_emebeds(algo)
    
    print ("Loading OOV entities")
    oov_entities = load_oov_entities(wiki_items)
    # find OOV entities in word embedding space, most_similar() needs the word to be existent in W.
    entity_label_found = search_in(word_embeds, oov_entities)
    print("Entities found with word embeddings: %d" % len(entity_label_found))
    
    # find the top n most similar words to the oov entity label
    oov_ent_embed = []
    oov_id_entity_map = {}
    oov_id = 0
    
    print ("Loading TransE")
    entity_id_map, label_entity_map, ent_embed = load_transe(wiki_items)
    del wiki_items # free memory
    
    for ent, label in entity_label_found.items():
        
        matched_embed = [] # of TransE
        topn_sim = word_embeds.most_similar([label], topn=config['topn_sim'])
        
        # chances are that at least one synonym of the missing embedding is in TransE
        for word in topn_sim:
            if word[0] in label_entity_map:
                matched_ent = label_entity_map[word[0]]
                matched_embed.append(ent_embed[entity_id_map[matched_ent]])
    
        if len(matched_embed) > 0:
            matched_embed = np.asarray(matched_embed)
            oov_ent_embed.append(np.average(matched_embed, axis=0))
            oov_id_entity_map[oov_id] = ent
            oov_id += 1
            
    np.save(os.path.join(config['transe_dir'], out+'.npy'), oov_ent_embed)
    to_pkl(oov_id_entity_map, os.path.join(config['transe_dir'], out+'.pickle'))
    
    print("Embeddings found by matching method: %d with topn: %d" %(len(oov_ent_embed), config['topn_sim']))


if __name__ == "__main__":
    
    plac.call(main)