import os
import numpy as np
#import gensim
import pickle as pkl
import json
import plac

#configure paths
config = {}
config["vocabs_dir"] = "vocabs"
config["wikidata_dir"] = "datasets/wikidata_dir"
config["embed_dir"] = "datasets/embed_dir"
config["transe_dim"] = 100

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
            else:
                # sentences of the form Word1_Word2
                words = label.split(" ")
                capitalized = []
                for word in words:
                    capitalized.append(word.capitalize())
        
                sentence = "_".join(capitalized)
    
                if sentence in word_embeds:
                    entity_label_found[ent] = sentence
                
    return entity_label_found


def load_word_embeds(file_name):
  
    print("Loading word embeds")
    
    f = open(file_name,'r')
    embeds = {}
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]], dtype='float32')
        embeds[word] = embedding
        
    print("Done.", len(embeds)," words loaded!")
    print('Done, %d words loaded.' % len(embeds))
    
    return embeds


@plac.annotations(
    embed=('Word embeddings of same dimension as TransE', 'positional', None, str),
    out=('Output file name', 'positional', None, str)
)
def main(embed, out):
    
    print ("Loading wiki items")
    wiki_items = load_wiki_items()
    
    print ("Loading word embeddings")
    word_embeds = load_word_embeds(embed)
    
    if len(list(word_embeds.values())[0]) != config['transe_dim']:
        exit('Word embeddings have diff. dimension as TransE embeddings')

    print ("Loading OOV entities")
    oov_entities = load_oov_entities(wiki_items)

    # find OOV entities in word embedding space.
    entity_label_found = search_in(word_embeds, oov_entities)
    
    # find the top n most similar words to the oov entity label
    oov_ent_embed = []
    oov_id_entity_map = {}
    oov_id = 0
    
    for ent, label in entity_label_found.items():
        
        oov_ent_embed.append(word_embeds[label])
        oov_id_entity_map[oov_id] = ent
        oov_id += 1

    np.save(os.path.join(config['transe_dir'], out+'_ent_embed.npy'), oov_ent_embed)
    to_pkl(oov_id_entity_map, os.path.join(config['transe_dir'], out+'_id_ent_map.pickle'))
    
    print("Embeddings found by matching method: %d" % len(oov_ent_embed))


if __name__ == "__main__":
    
    plac.call(main)
