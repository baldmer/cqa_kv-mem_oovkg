
# process all cqa dataset to extract simple direct cqa data.

import os
from io import open


root_path = "datasets/preprocessed_data_full_cqa/valid/"
out_root_path = "my_datasets/preprocessed_data_simple_cqa/valid"

for root, dirs, files in os.walk(root_path):

    #qa_file_name = "train_simple_direct_qa_pairs_qids.txt" 
    
    for name in dirs:
        
        states_path = os.path.join(root, name, name+'_state.txt')
        states = open(states_path, "r", encoding='utf-8').read().strip().split('\n')
        
        response_ints_path = os.path.join(root, name, name+'_response_ints.txt')
        response_ints = open(response_ints_path, "r", encoding='utf-8').read().split('\n')
        
        response_entities_path = os.path.join(root, name, name+'_response_entities.txt')
        response_entities = open(response_entities_path, "r", encoding='utf-8').read().split('\n')
        
        response_bools_path = os.path.join(root, name, name+'_response_bools.txt')
        response_bools = open(response_bools_path, "r", encoding='utf-8').read().split('\n')
        
        orig_response_path = os.path.join(root, name, name+'_orig_response.txt')
        orig_response = open(orig_response_path, "r", encoding='utf-8').read().split('\n')
        
        context_utterance_path = os.path.join(root, name, name+'_context_utterance.txt')
        context_utterance = open(context_utterance_path, "r", encoding='utf-8').read().split('\n')
        
        context_types_path = os.path.join(root, name, name+'_context_types.txt')
        context_types = open(context_types_path, "r", encoding='utf-8').read().split('\n')
        
        context_relations_path = os.path.join(root, name, name+'_context_relations.txt')
        context_relations = open(context_relations_path, "r", encoding='utf-8').read().split('\n')
        
        context_ints_path = os.path.join(root, name, name+'_context_ints.txt')
        context_ints = open(context_ints_path, "r", encoding='utf-8').read().split('\n')
        
        context_entities_path = os.path.join(root, name, name+'_context_entities.txt')
        context_entities = open(context_entities_path, "r", encoding='utf-8').read().split('\n')
        
        context_path = os.path.join(root, name, name+'_context.txt')
        context = open(context_path, "r", encoding='utf-8').read().split('\n')
        
        # create directory with "name"
        out_path = os.path.join(out_root_path, name)
        os.mkdir(out_path)
        
        # open new files in append mode.
        
        out_state = os.path.join(out_path, name+'_state.txt')
        out_response_ints = os.path.join(out_path, name+'_response_ints.txt')
        out_response_entities = os.path.join(out_path, name+'_response_entities.txt')
        out_response_bools = os.path.join(out_path, name+'_response_bools.txt')
        out_orig_response = os.path.join(out_path, name+'_orig_response.txt')
        out_context_utterance = os.path.join(out_path, name+'_context_utterance.txt')
        out_context_types = os.path.join(out_path, name+'_context_types.txt')
        out_context_relations = os.path.join(out_path, name+'_context_relations.txt')
        out_context_ints = os.path.join(out_path, name+'_context_ints.txt')
        out_context_entities = os.path.join(out_path, name+'_context_entities.txt')
        out_context = os.path.join(out_path, name+'_context.txt')
        
        with open(out_state, 'a') as out_state_f, open(out_response_ints, 'a') as out_response_ints_f, open(out_response_entities, 'a') as out_response_entities_f, open(out_response_bools, 'a') as out_response_bools_f, open(out_orig_response, 'a') as out_orig_response_f, open(out_context_utterance, 'a') as out_context_utterance_f, open(out_context_types, 'a') as out_context_types_f, open(out_context_relations, 'a') as out_context_relations_f, open(out_context_ints, 'a') as out_context_ints_f, open(out_context_entities, 'a') as out_context_entities_f, open(out_context, 'a') as out_context_f:
            
            for i in range(len(states)):
                state_i = states[i].strip()
                if state_i == "Simple Question (Direct)": 
                    out_state_f.write(state_i+'\n')
                    out_response_ints_f.write(response_ints[i]+'\n')
                    out_response_entities_f.write(response_entities[i]+'\n')
                    out_response_bools_f.write(response_bools[i]+'\n')
                    out_orig_response_f.write(orig_response[i]+'\n')
                    out_context_utterance_f.write(context_utterance[i]+'\n')
                    out_context_types_f.write(context_types[i]+'\n')
                    out_context_relations_f.write(context_relations[i]+'\n')
                    out_context_ints_f.write(context_ints[i]+'\n')
                    out_context_entities_f.write(context_entities[i]+'\n')
                    out_context_f.write(context[i]+'\n')
                    
    
    