""" based on baseline, can be found at https://github.com/amritasaha1812/CSQA_Code/"""
import os
import random
import pickle as pkl
import json
import plac

SEED = 1234
random.seed(SEED)

#from global ids
pad_kb_symbol_index = 0
pad_kb_symbol = '<pad_kb>'
nkb_symbol_index = 1
nkb_symbol = '<nkb>'

NUM_TRANSE_EMBED = 9274339
MAX_CANDIDATE_ENTITIES = 10

# configure paths here    
config = {}
config['wikidata_dir'] = "datasets/wikidata_dir"
config['transe_dir'] = "datasets/transe_dir"
config['use_baseline_algo'] = False
config['max_mem_size'] = 10 # None, no class balance, get the num. of q. with no cand.
# file containing oov embeds., we need id_ent_map only, assumed to be in transe_dir
#config['oov_handler'] = ''

def extract_dimension_from_tuples_as_list(list_of_tuples, dim):
    result = []
    for tup in list_of_tuples:
        result.append(tup[dim])

    return result


PIPE = "|"
def get_str_of_seq(entities):
    return PIPE.join(entities)


def pad_or_clip_memory(tuples):
    '''pad or clip to max memory size'''
    
    tuples = list(tuples)
    if len(tuples) > config['max_mem_size']:
        tuples = tuples[:config['max_mem_size']]
    elif len(tuples) < config['max_mem_size']:
        pad_len = config['max_mem_size'] - len(tuples)
        nkb_tuple = (nkb_symbol, nkb_symbol, nkb_symbol)
        tuples = tuples + [nkb_tuple] * pad_len
    
    random.shuffle(tuples)

    return tuples


def load_wikidata(dir_name):
    
    if not os.path.exists(dir_name):
        sys.exit('Wikidata path do not exists.')
    
    with open(dir_name+'/wikidata_short_1_reduced.json', encoding='utf-8') as data_file:
        wikidata = json.load(data_file)
        
    print('Successfully loaded wikidata1')

    with open(dir_name+'/wikidata_short_2_reduced.json', encoding='utf-8') as data_file:
        wikidata2 = json.load(data_file)
        
    print('Successfully loaded wikidata2')

    wikidata.update(wikidata2)
    
    # free memory
    del wikidata2

    # total: 13,987,819 id:label pairs
    with open(dir_name+'/items_wikidata_n.json', encoding='utf-8') as data_file:
        item_data = json.load(data_file)
    
    print('Successfully loaded items json')
        
    wikidata_remove_list = [q for q in wikidata if q not in item_data]

    # free memory
    del item_data

    #P31: is deprecated use P2868 or P3831 instead
    wikidata_remove_list.extend([q for q in wikidata if 'P31' not in wikidata[q] and 'P279' not in wikidata[q]])
    wikidata_remove_list.extend([u'Q7375063', u'Q24284139', u'Q1892495', u'Q22980687', u'Q25093915', u'Q22980685', 
                                 u'Q22980688', u'Q25588222', u'Q1668023', u'Q20794889', u'Q22980686',u'Q297106',u'Q1293664'])
    # wikidata_remove_list.extend([q for q in wikidata if q not in child_par_dict])

    for q in wikidata_remove_list:
        wikidata.pop(q,None)
    
    print ("%s entities removed from wikidata" % len(wikidata_remove_list))
    
    # free memory
    del wikidata_remove_list
    
    with open(dir_name+'/comp_wikidata_rev.json', encoding='utf-8') as data_file:
        reverse_dict = json.load(data_file)
        
    print('Successfully loaded reverse_dict json')
    
    # all entities in wikidata with their parent (which may be the 1-hop or 2-hop parent of the entity) 9,035,990 pairs.
    with open(dir_name+'/child_par_dict_save.json', encoding='utf-8') as data_file:
        child_par_dict = json.load(data_file)
        
    print('Successfully loaded child_par_dict save json')
    
    with open(dir_name+'/child_all_parents_till_5_levels.json', encoding='utf-8') as data_file:
        child_all_par_dict = json.load(data_file)
        
    print('Successfully loaded child_all_parents_dict_till_5_levels json')

    # total 569 relation:label pairs
    with open(dir_name+'/filtered_property_wikidata4.json', encoding='utf-8') as data_file:
        prop_data = json.load(data_file)
    
    print('Successfully loaded filtered_property_wikidata4 json')
    
    with open(dir_name+'/par_child_dict.json', encoding='utf-8') as f1:
        par_child_dict = json.load(f1)

    print('Successfully loaded par_child_dict json')
    
    # all entities in wikidata with their immediate parent. total: 12,773,652 child:par pairs.
    with open(dir_name+'/child_par_dict_immed.json', encoding='utf-8') as data_file:
    	child_par_dict_immed = json.load(data_file)
    	
    # Fix for parent types (wikimedia, metaclass, etc.), refer to https://github.com/amritasaha1812/CSQA_Code/
    stop_par_list = ['Q21025364', 'Q19361238', 'Q21027609', 'Q20088085', 'Q15184295', 'Q11266439',
                     'Q17362920', 'Q19798645', 'Q26884324', 'Q14204246', 'Q13406463', 'Q14827288',
                     'Q4167410', 'Q21484471', 'Q17442446', 'Q4167836', 'Q19478619', 'Q24017414', 
                     'Q19361238', 'Q24027526', 'Q15831596', 'Q24027474', 'Q23958852', 'Q24017465', 
                     'Q24027515', 'Q1924819']
    
    stop_par_immed_list = ['Q10876391', 'Q1351452', 'Q1423994', 'Q1443451', 'Q14943910', 'Q151', 
                           'Q15156455', 'Q15214930', 'Q15407973', 'Q15647814', 'Q15671253', 'Q162032', 
                           'Q16222597', 'Q17146139', 'Q17633526', 'Q19798644', 'Q19826567', 'Q19842659', 
                           'Q19887878', 'Q20010800', 'Q20113609', 'Q20116696', 'Q20671729', 'Q20769160', 
                           'Q20769287', 'Q21281405', 'Q21286738', 'Q21450877', 'Q21469493', 'Q21705225', 
                           'Q22001316', 'Q22001389', 'Q22001390', 'Q23840898', 'Q23894246', 'Q24025936', 
                           'Q24046192', 'Q24571886', 'Q24731821', 'Q2492014', 'Q252944', 'Q26267864', 
                           'Q35120', 'Q351749', 'Q367', 'Q370', 'Q3933727', 'Q4663903', 'Q4989363', 
                           'Q52', 'Q5296', 'Q565', 'Q6540697', 'Q79786', 'Q964']

    ent_list = []

    for x in stop_par_list:
         ent_list.extend(par_child_dict[x])

    ent_list = list(set(ent_list))
    
    ent_list_resolved = [x for x in ent_list if x in child_par_dict_immed and child_par_dict_immed[x] not in stop_par_list and child_par_dict_immed[x] not in stop_par_immed_list]

    child_par_dict_val = list(set(child_par_dict.values()))
    old_2_new_pars_map = {x:x for x in child_par_dict_val}
    rem_par_list = set()

    for x in ent_list_resolved:
         child_par_dict[x] = child_par_dict_immed[x]
         old_2_new_pars_map[child_par_dict[x]] = child_par_dict_immed[x]
         rem_par_list.add(child_par_dict[x])

    ent_list_discard = list(set(ent_list) - set(ent_list_resolved))

    for q in ent_list_discard:
         par_q = None
         if q in child_par_dict:
             child_par_dict.pop(q, None)
         if q in wikidata:
             wikidata.pop(q,None)
         if q in reverse_dict:
             reverse_dict.pop(q, None)
        
    # free memory
    del child_par_dict_immed
    del par_child_dict

    return wikidata, reverse_dict, prop_data, child_par_dict, child_all_par_dict
    
#TODO: specify oov handling as config paramters
def load_transe_data(dir_name, oov_id_ent_map):
    
    if not os.path.exists(dir_name):
        sys.exit('TransE path do not exists.')
    
    print("Reading transe data")
        
    id_entity_map = {pad_kb_symbol_index:pad_kb_symbol, nkb_symbol_index: nkb_symbol}
    id_entity_map.update({(k+2):v for k,v in pkl.load(open(dir_name+'/id_ent_map.pickle','rb')).items()})
    
    if oov_id_ent_map:
        try:
            id_entity_map.update({(k+2+NUM_TRANSE_EMBED):v for k,v in pkl.load(open(oov_id_ent_map,'rb')).items()})
        except:
            exit('Incorrect oov file name')

    entity_id_map = {v: k for k, v in id_entity_map.items()}

    id_rel_map = {pad_kb_symbol_index:pad_kb_symbol, nkb_symbol_index: nkb_symbol}
    id_rel_map.update({(k+2):v for k,v in pkl.load(open(dir_name+'/id_rel_map.pickle','rb')).items()})
    rel_id_map = {v: k for k, v in id_rel_map.items()}

    return id_entity_map, entity_id_map, id_rel_map, rel_id_map


def get_tuples_involving_entities(candidate_entities, all_wikidata, transe_data, relations_in_context=None, types_in_context=None):
    tuples = set([])
    #tuples = []
    #rev_tuples = set([])
    pids = set([])
    
    wikidata, reverse_dict, prop_data, child_par_dict, child_all_par_dict = all_wikidata
    _, entity_id_map, _, rel_id_map = transe_data
    
    cand = [q1 for q1 in candidate_entities if q1 in child_par_dict and q1 in entity_id_map]
    rev_cand = [q1 for q1 in candidate_entities if q1 in reverse_dict and q1 in entity_id_map]
    
    cand = set(cand).union(set(rev_cand))
    
    for QID in cand:
        
        detected_pids = set()
        rev_feasible_pids = set()
        #feasible_pids = set()
        
        wiki_feasible_pids = set()
        #search relations
        
        if QID in wikidata: # and not in child_par_dict
            wiki_feasible_pids = [p for p in wikidata[QID] if p in prop_data and p in rel_id_map]
        '''
        if QID in child_par_dict:
            #feasible_pids = wiki_feasible_pids
            feasible_pids = wiki_feasible_pids
        '''
        if QID in reverse_dict:
            rev_feasible_pids = [p for p in reverse_dict[QID] if p in prop_data and p in rel_id_map]
            
        if relations_in_context is not None:
            #detected_pids = set(feasible_pids).intersection(relations_in_context)
            detected_pids = set(wiki_feasible_pids).intersection(relations_in_context)
            if len(detected_pids) == 0:
                detected_pids = set(rev_feasible_pids).intersection(relations_in_context)
            '''    
            if len(detected_pids) == 0:
                #instead of using all pids, we use only the ones for the relation in wikidata
                detected_pids = set(wiki_feasible_pids).intersection(relations_in_context)
            '''
        pids.update(detected_pids)
        
        wiki_feasible_qids = set()
        feasible_qids = set()
        rev_feasible_qids = set()
        
        #get objects/the answer, based on the relations found
        for pid in detected_pids:
            
            # first valid. ensure QID is from respective set.
            
            if QID in wikidata and pid in wikidata[QID]:
                wiki_feasible_qids = set([q for q in wikidata[QID][pid] if q in entity_id_map and q in wikidata])
            
            if QID in child_par_dict and pid in wikidata[QID]:
                feasible_qids = set([q for q in wikidata[QID][pid] if q in entity_id_map and q in child_par_dict])
            
            if QID in reverse_dict and pid in reverse_dict[QID]:
                rev_feasible_qids = set([q for q in reverse_dict[QID][pid] if q in entity_id_map and q in child_par_dict])
            
            detected_qids = set([x for x in feasible_qids if len(set(child_all_par_dict[x]).intersection(types_in_context))>0])
                
            if len(detected_qids) == 0:
                # search in rev.
                detected_qids = set([x for x in rev_feasible_qids if len(set(child_all_par_dict[x]).intersection(types_in_context))>0])
                    
            if len(detected_qids) == 0:
                detected_qids = wiki_feasible_qids.union(feasible_qids).union(rev_feasible_qids)
               
            for qid in detected_qids:
                tuples.add((QID, pid, qid))
                tuples.add((qid, pid, QID))   
    
    # we keep the same order on every execution.
    tuples = sorted(tuples)
    
    return tuples, pids


# from baseline for comparative: https://github.com/amritasaha1812/CSQA_Code/
def get_tuples_involving_entities_base(candidate_entities, all_wikidata, transe_data, relations_in_context=None, types_in_context=None):
    tuples = set([])
    pids = set([])
    
    wikidata, reverse_dict, prop_data, child_par_dict, child_all_par_dict = all_wikidata
    _, entity_id_map, _, rel_id_map = transe_data
    
    for QID in [q1 for q1 in candidate_entities if q1 in child_par_dict and q1 in entity_id_map]:
        QID_type_matched = False
        if types_in_context is None or (QID in child_all_par_dict and len(set(child_all_par_dict[QID]).intersection(types_in_context))>0):
                QID_type_matched = True
        feasible_pids = [p for p in wikidata[QID] if p in prop_data and p in rel_id_map]
        if relations_in_context is not None:
                detected_pids = set(feasible_pids).intersection(relations_in_context)
                if len(detected_pids)==0:
                        detected_pids = set(feasible_pids)
        else:
                detected_pids = set(feasible_pids)
        pids.update(detected_pids)
        for pid in detected_pids:
            feasible_qids = set([q for q in wikidata[QID][pid] if q in entity_id_map and q in child_par_dict])
            if types_in_context is None or QID_type_matched:
                detected_qids = feasible_qids
            else:
                detected_qids = set([x for x in feasible_qids if len(set(child_all_par_dict[x]).intersection(types_in_context))>0])
            if len(detected_qids)==0:
                detected_qids = feasible_qids
            for qid in detected_qids:
                tuples.add((QID, pid, qid))
                tuples.add((qid, pid, QID))    
    for QID in [q1 for q1 in candidate_entities if q1 in reverse_dict and q1 in entity_id_map]:
        QID_type_matched = False
        if types_in_context is None or (QID in child_all_par_dict and len(set(child_all_par_dict[QID]).intersection(types_in_context))>0):
                QID_type_matched = True
        feasible_pids = [p for p in reverse_dict[QID] if p in prop_data and p in rel_id_map]
        if relations_in_context is not None:
                detected_pids = set(feasible_pids).intersection(relations_in_context)
                if len(detected_pids)==0:
                        detected_pids = set(feasible_pids)
        else:
                detected_pids = set(feasible_pids)
        pids.update(detected_pids)
        for pid in detected_pids:
            feasible_qids = set([q for q in reverse_dict[QID][pid] if q in entity_id_map and q in child_par_dict])
            if types_in_context is None or QID_type_matched:
                detected_qids = feasible_qids
            else:
                detected_qids = set([x for x in feasible_qids if len(set(child_all_par_dict[x]).intersection(types_in_context))>0])
            if len(detected_qids)==0:
                detected_qids = feasible_qids
            for qid in detected_qids:#[q for q in wikidata[QID][pid] if q in entity_id_map]:
                tuples.add((QID, pid, qid))
                tuples.add((qid, pid, QID))

    return tuples, pids


# TODO: data splits can be processed all at once, similarly to extract_simple_cqa.py

@plac.annotations(
    corpus_path=('Path to the corpus dataset (split)', 'positional', None, str),
    oov_ent_map=('File containing oov embeds. (id_ent_map)', 'option', 'oov', str)
)
def main(corpus_path, oov_ent_map=None):
    
    if not os.path.exists(corpus_path):
        sys.exit('Dataset path do not exists.')
   
    wikidata = load_wikidata(config["wikidata_dir"])
    transe_data = load_transe_data(config["transe_dir"], oov_ent_map)
    
    max_mem_size = 0
    total_sub_candidates = 0
    total_oov = 0
    num_no_mem_cand = 0
    num_mem_cand = 0
    num_correct_tuples = 0
    

    for root, dirs, files in os.walk(corpus_path):        
        for dir_name in dirs:
            
            #safe way to consider all questions.
            states_path = os.path.join(root, dir_name, dir_name+'_state.txt')
            states = open(states_path, encoding='utf-8').read().strip().split('\n')

            context_entities_path = os.path.join(root, dir_name, dir_name+'_context_entities.txt')
            context_entities = open(context_entities_path, encoding='utf-8').read().split('\n')
            
            context_relations_path = os.path.join(root, dir_name, dir_name+'_context_relations.txt')
            context_relations = open(context_relations_path, encoding='utf-8').read().split('\n')
            
            context_types_path = os.path.join(root, dir_name, dir_name+'_context_types.txt')
            context_types = open(context_types_path, encoding='utf-8').read().split('\n')
            
            # THIS IS WHAT WE TRY TO FIND
            response_entities_path = os.path.join(root, dir_name, dir_name+'_response_entities.txt')
            response_entities = open(response_entities_path, encoding='utf-8').read().split('\n')
            
            out_sources = os.path.join(corpus_path, dir_name, dir_name+'_sources.txt')
            out_relations = os.path.join(corpus_path, dir_name, dir_name+'_relations.txt')
            out_targets = os.path.join(corpus_path, dir_name, dir_name+'_key_targets.txt')
            
            out_cand_stats = os.path.join(corpus_path, dir_name, dir_name+'_candidates_stats.txt')
            
            print ("processing dir %s" % dir_name)
            
            with open(out_sources, "w") as out_sources_f, open(out_relations, "w") as out_relations_f, open(out_targets, "w") as out_targets_f, open(out_cand_stats, "w") as out_cand_stats_f:
                
                for i in range(len(states)):
                    
                    candidate_entities = context_entities[i].strip().split("|")
                    relations_in_context = set(context_relations[i].strip().split("|"))
                    types_in_context = set(context_types[i].strip().split("|"))
                    
                    resp_entities = response_entities[i].strip().split("|")
                    
                    if len(candidate_entities) > MAX_CANDIDATE_ENTITIES:
                        candidate_entities = candidate_entities[:MAX_CANDIDATE_ENTITIES]
                    
                    if config['use_baseline_algo']:
                        tuples, relations_explored = get_tuples_involving_entities_base(candidate_entities, wikidata, transe_data, relations_in_context, types_in_context)
                    else:
                        tuples, relations_explored = get_tuples_involving_entities(candidate_entities, wikidata, transe_data, relations_in_context, types_in_context)
                        
                    if config['max_mem_size'] is not None:
                        tuples = pad_or_clip_memory(tuples)
                    
                    '''
                    total_oov += oov_cand_num
                    total_sub_candidates += len(candidate_entities)
                    '''
                    
                    if len(tuples) == 0:
                        num_no_mem_cand += 1
                    else:
                        num_mem_cand += 1                    
    
                    #print ("Triples processed: %d " % len(tuples))
                        
                    '''
                    for cand in candidate_entities:
                        if [cand, list(relations_in_context)[0], resp_entities[0]] in tuples:
                            num_correct_tuples += 1
                            break
                    '''
                    
                    sources = extract_dimension_from_tuples_as_list(tuples, 0)
                    relations = extract_dimension_from_tuples_as_list(tuples, 1)
                    targets = extract_dimension_from_tuples_as_list(tuples, 2)
            
                    # write strings to files
                    
                    out_sources_f.write(get_str_of_seq(sources)+'\n')
                    out_relations_f.write(get_str_of_seq(relations)+'\n')
                    out_targets_f.write(get_str_of_seq(targets)+'\n')
                    
                    out_cand_stats_f.write("%s, %d, %d, %d, %d \n" %(dir_name, len(tuples), len(sources), len(relations), len(targets)))
                    
                    if len(tuples) > max_mem_size:
                        max_mem_size = len(tuples) 
        
    print("Max. memory size needed: %s" % max_mem_size)
    
    #print("Number of subject candidates: %s" % total_sub_candidates) #could be repeated
    #print("Number of OOV subject candidates (no embedding): %s" % total_oov)
    
    print("Number of questions with no memory candidates: %d - and mem-cands %d" % (num_no_mem_cand, num_mem_cand))
    
    #print("Number of correct tuples with memory candidates: %d" % num_correct_tuples)

                
if __name__ == "__main__":
    plac.call(main)
    
    
    
    
