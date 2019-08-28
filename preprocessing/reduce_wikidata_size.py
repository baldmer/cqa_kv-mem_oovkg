import os
from io import open
import pickle as pkl

import json

'''
only entities in subject vocab are used, those are used to generate the
candidates for the kvmm, e.g. only wikidata[QID] is checked, QID is
for the set of subject entities.
'''

def load_wikidata(dir_name):
    with open(dir_name+'/wikidata_short_1.json', encoding='utf-8') as f:
        wikidata = json.load(f)
        
    print('Successfully loaded %s '%dir_name)

    return wikidata


def reduce_size(wikidata, subject_vocab):
    '''return a reduced version of wikidata, based on entities present in vocab.'''
    
    return {q:v for q,v in wikidata.items() if q in subject_vocab}

                
def main():
    
    subject_vocab = pkl.load(open("vocabs/subject_vocab.pkl", "rb"))
    
    #process wikidata1
    wikidata = load_wikidata("datasets/wikidata_dir/wikidata_short_1.json")
    wikidata = reduce_size(wikidata, subject_vocab)
    
    with open('my_datasets/wikidata_dir/wikidata_short_1.json', 'w', encoding='utf-8') as f:
        json.dump(wikidata, f)
        
    #process wikidata2
    wikidata = load_wikidata("datasets/wikidata_dir/wikidata_short_2.json")
    wikidata = reduce_size(wikidata, subject_vocab)
    
    with open('my_datasets/wikidata_dir/wikidata_short_2.json', 'w', encoding='utf-8') as f:
        json.dump(wikidata, f)
    

if __name__ == "__main__":
    main()

    
    