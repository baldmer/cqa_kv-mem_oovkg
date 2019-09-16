# extract simple cqa
python preprocessing/extract_simple_cqa.py datasets/preprocessed_data_full_cqa/ datasets/simple_cqa_dataset_no_oov_handling_new_mem/
# extract memory candidates
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/train
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/valid
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/test
# binarize corpus
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/train datasets/no_oov_handling_new_mem/train.pkl
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/valid datasets/no_oov_handling_new_mem/valid.pkl
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_new_mem/test datasets/no_oov_handling_new_mem/test.pkl

