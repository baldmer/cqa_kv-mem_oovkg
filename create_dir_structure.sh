#!/bin/bash

datasets="datasets"
[ ! -d "$datasets" ] && mkdir -p "$datasets"

transe_dir="datasets/transe_dir"
[ ! -d "$transe_dir" ] && mkdir -p "$transe_dir"

wikidata_dir="datasets/wikidata_dir"
[ ! -d "$wikidata_dir" ] && mkdir -p "$wikidata_dir"

embed_dir="datasets/embed_dir"
[ ! -d "$embed_dir" ] && mkdir -p "$embed_dir"

vocabs_dir="vocabs"
[ ! -d "$vocabs_dir" ] && mkdir -p "$vocabs_dir"

# where final pre-processed dataset files are created

newdir="datasets/no_oov_handling_base_mem"
[ ! -d "$newdir" ] && mkdir -p "$newdir"

newdir="datasets/no_oov_handling_new_mem"
[ ! -d "$newdir" ] && mkdir -p "$newdir"

newdir="datasets/oov_handling_matching"
[ ! -d "$newdir" ] && mkdir -p "$newdir"

newdir="datasets/oov_handling_pca"
[ ! -d "$newdir" ] && mkdir -p "$newdir"

# extracted simple cqa datasets are created by extraction script.
 
