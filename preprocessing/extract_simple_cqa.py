
# process all cqa dataset to extract simple direct cqa data.

import os
from io import open
import plac


@plac.annotations(
    in_dir=('Dataset for preprocessing', 'positional', None, str),
    out_dir=('Output dataset directory', 'positional', None, str)
)
def main(in_dir, out_dir):

    data_dirs = os.listdir(in_dir)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for data_dir in data_dirs:
        
        q_dirs = os.listdir(os.path.join(in_dir, data_dir))
        os.mkdir(os.path.join(out_dir, data_dir))
        
        for q_dir in q_dirs:
            
            states_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_state.txt')
            states = open(states_path, "r", encoding='utf-8').read().strip().split('\n')
            
            response_ints_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_response_ints.txt')
            response_ints = open(response_ints_path, "r", encoding='utf-8').read().split('\n')
            
            response_entities_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_response_entities.txt')
            response_entities = open(response_entities_path, "r", encoding='utf-8').read().split('\n')
            
            response_bools_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_response_bools.txt')
            response_bools = open(response_bools_path, "r", encoding='utf-8').read().split('\n')
            
            orig_response_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_orig_response.txt')
            orig_response = open(orig_response_path, "r", encoding='utf-8').read().split('\n')
            
            context_utterance_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context_utterance.txt')
            context_utterance = open(context_utterance_path, "r", encoding='utf-8').read().split('\n')
            
            context_types_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context_types.txt')
            context_types = open(context_types_path, "r", encoding='utf-8').read().split('\n')
            
            context_relations_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context_relations.txt')
            context_relations = open(context_relations_path, "r", encoding='utf-8').read().split('\n')
            
            context_ints_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context_ints.txt')
            context_ints = open(context_ints_path, "r", encoding='utf-8').read().split('\n')
            
            context_entities_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context_entities.txt')
            context_entities = open(context_entities_path, "r", encoding='utf-8').read().split('\n')
            
            context_path = os.path.join(in_dir, data_dir, q_dir, q_dir+'_context.txt')
            context = open(context_path, "r", encoding='utf-8').read().split('\n')
            
            # create directory with "name"
            out_path = os.path.join(out_dir, data_dir, q_dir)
            os.mkdir(out_path)
            
            # open new files in append mode.
            
            out_state = os.path.join(out_dir, data_dir, q_dir, q_dir+'_state.txt')
            out_response_ints = os.path.join(out_dir, data_dir, q_dir, q_dir+'_response_ints.txt')
            out_response_entities = os.path.join(out_dir, data_dir, q_dir, q_dir+'_response_entities.txt')
            out_response_bools = os.path.join(out_dir, data_dir, q_dir, q_dir+'_response_bools.txt')
            out_orig_response = os.path.join(out_dir, data_dir, q_dir, q_dir+'_orig_response.txt')
            out_context_utterance = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context_utterance.txt')
            out_context_types = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context_types.txt')
            out_context_relations = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context_relations.txt')
            out_context_ints = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context_ints.txt')
            out_context_entities = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context_entities.txt')
            out_context = os.path.join(out_dir, data_dir, q_dir, q_dir+'_context.txt')
            
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
                        

if __name__ == "__main__":
    plac.call(main)
