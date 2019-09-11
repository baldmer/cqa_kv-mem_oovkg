import os
from io import open
import pickle as pkl
import json


def load_wikidata(file_name):
    with open(file_name, encoding='utf-8') as f:
        wikidata = json.load(f)
        
    print('Successfully loaded %s '% file_name)

    return wikidata


def reduce_size(wikidata, ents_vocab, ents_oov):
    '''return a reduced version of wikidata, based on entities present in vocab.'''
    
    return {q:v for q,v in wikidata.items() if q in ents_vocab or q in ents_oov}

                
def main():
    
    ents_vocab = pkl.load(open("../vocabs/entities_vocab.pkl", "rb"))
    ents_oov = pkl.load(open("../vocabs/entities_oov.pkl", "rb"))
    
    #process wikidata1
    wikidata = load_wikidata("../datasets/wikidata_dir/wikidata_short_1.json")
    wikidata = reduce_size(wikidata, ents_vocab, ents_oov)
    
    with open('../datasets/wikidata_dir/wikidata_short_1_reduced.json', 'w', encoding='utf-8') as f:
        json.dump(wikidata, f)
        
    #process wikidata2
    wikidata = load_wikidata("../datasets/wikidata_dir/wikidata_short_2.json")
    wikidata = reduce_size(wikidata, ents_vocab, ents_oov)
    
    with open('../datasets/wikidata_dir/wikidata_short_2_reduced.json', 'w', encoding='utf-8') as f:
        json.dump(wikidata, f)
    

if __name__ == "__main__":
    main()

    
    