#!/bin/bash

datasets="datasets"
[ ! -d "$datasets" ] && mkdir -p "$datasets"

transe_dir="datasets/transe_dir"
[ ! -d "$transe_dir" ] && mkdir -p "$transe_dir"

wikidata_dir="datasets/wikidata_dir"
[ ! -d "$wikidata_dir" ] && mkdir -p "$wikidata_dir"


#no_oov_handling_base_mem  oov_handling_matching  preprocessed_data_full_cqa  transe_dir
#no_oov_handling_new_mem   oov_handling_pca       simple_cqa_dataset          wikidata_dir

