
# Install requirements

```
$ pip install -r requirements.txt
```

Might need to download NLTK resources, e.g.:

```
import nltk
nltk.download('punkt')
```

## Install PyTorch

### No CUDA

```
pip3 install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### CUDA 9

```
pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

For more information refer to the [PyTorch](https://pytorch.org/) we page.

# Setup

Run the following script to create a initial directory structure.

```
sh create_dir_structure.sh
```

Download the public [CQA dataset](https://amritasaha1812.github.io/CSQA/download_CQA/), place the folder `preprocessed_data_full_cqa` into the `datasets` directory.


Download the pre-trained [TransE](https://drive.google.com/file/d/1AD_7xesdcJEtth6SZdF5xTllRqPZ3E6-/view?usp=sharing) [1] embeddings and place its content in `datasets/transe_dir`.

Download the [pre-processed wikidata](https://amritasaha1812.github.io/CSQA/download/) and move its content to `datasets/wikidata_dir`.

The pre-trained word embeddings ([word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) [2,3]) are placed in `datasets/embed_dir`.

For more information on data sets refer to [CSQA](https://amritasaha1812.github.io/CSQA/) [4].

## Generate offline OOVKG approaches

### PCA approach

Reduce dimension of pre-trained word model.

```
python oov_approaches/oov_pca_reduce_dim.py
```
To generate the OOVKG embeddings with the PCA approach, execute the following script with the help option.

```
python oov_approaches/oov_pca.py -h
```
### Text-Match approach

To generate the OOVKG embeddings with the Text-Match approach, execute the following script with the help option.

```
python oov_approaches/oov_matching_label -h
```
# Data set Pre-processing steps

Create vocabularies

```
python preprocessing/create_kg_vocabs.py datasets/simple_cqa_dataset/ datasets/transe_dir/ vocabs/
```

Reduce Wikidata size (optional - if having RAM issues)

```
python preprocessing/reduce_wikidata_size.py
```

Extract memory candidates and produce the final data sets.
```
sh preprocess_no_oov_handling_new_mem.sh
sh preprocess_oov_handling_matching.sh
sh preprocess_oov_handling_pca.sh
```

# Training

The configuration parameters are specified in the `config` dictionary. The most important are specified as follows:

```
'save_name_prefix': 'oov_matching',
'train_data_file': "datasets/oov_handling_matching/train.pkl",
'test_data_file': "datasets/oov_handling_matching/test.pkl",
'valid_data_file': "datasets/oov_handling_matching/valid.pkl",
'oov_ent_handler': "oov_text_matching_ent_embed.npy",
```

Execute:

```
python kvmn.py train
```

To train the multi-answer prediction model execute:

```
python kvmn_multi.py train
```


# Testing

The configuration parameters are left as the same for training.

Execute:

```
python kvmn.py -mdl models/<MODEL_NAME.pt> test
```
To train the multi-answer prediction model execute:
```
python kvmn_multi.py -mdl models/<MODEL_NAME.pt> test
```
The result will be dumped to the `metrics/` folder.

# Evaluating

Execute

```
python metrics/prec_recall_f1.py metrics/<NAME_MODEL_TEST_OUTPUT.txt>
```
To evaluate the mult-answer prediction execute:
```
python metrics/prec_recall_f1_multi_target.py metrics/<NAME_MODEL_TEST_OUTPUT.txt>
```

# References

[1] Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. Translating embeddings for modeling multi-relational data. In Advances in neural information processing systems, pages 2787–2795, 2013.

[2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013.

[3] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems, pages 3111–3119, 2013.

[4] Amrita Saha, Vardaan Pahuja, Mitesh M Khapra, Karthik Sankaranarayanan, and Sarath Chandar. Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.






