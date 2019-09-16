# extract simple cqa
python preprocessing/extract_simple_cqa.py datasets/preprocessed_data_full_cqa/ datasets/simple_cqa_dataset_no_oov_handling_base_mem/
# extract memory candidates
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/train
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/valid
python preprocessing/create_kvmn_candidates.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/test
# binarize corpus
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/train datasets/no_oov_handling_base_mem/train.pkl
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/valid datasets/no_oov_handling_base_mem/valid.pkl
python preprocessing/binarize_corpus.py datasets/simple_cqa_dataset_no_oov_handling_base_mem/test datasets/no_oov_handling_base_mem/test.pkl

