import numpy as np
from sklearn.decomposition import PCA
import gensim

print ("Loading word embeddings...")

embeds = {}
embeds = gensim.models.KeyedVectors.load_word2vec_format('datasets/embed_dir/GoogleNews-vectors-negative300.bin', binary=True)

print("Done loading embeddings.")

W = []
W_labels = []

for idx, label in enumerate(embeds.wv.vocab):
    W.append(embeds[label])
    W_labels.append(label)

print (len(W))

W = np.asarray(W, dtype=np.float32)

def PPA(W, N, T=7):
    # substract mean
    W = W - np.mean(W)
    # calculate PCA components
    pca = PCA(n_components = N)
    W_fit = pca.fit_transform(W)
    U = pca.components_
    
    # substract the top D components
    for i, w in enumerate(W):
        for u in U[0:T]:
            W[i] = W[i] - np.dot(u.transpose(), W[i]) * u
    
    return W

# apply post-processing
print ("Post-processing")
T = 7
d = 300
W_prime = PPA(W, d, T)

print ("PCA dim. reduction")
# PCA Dim Reduction
# Use PCA to transform W_prime
d_prime = 100
W_prime = W_prime - np.mean(W_prime)
pca = PCA(n_components = d_prime)
W_prime = pca.fit_transform(W_prime)

# apply post-processing
print ("Post-processing again")
W_prime = PPA(W_prime, d_prime, T)

print ("Saving file...")

embed_file = open('GoogleNews-vectors-negative100_test2.txt', 'w')

for i, label in enumerate(W_labels):
    embed_file.write("%s\t" % label)
    
    for w_i in W_prime[i]:
        embed_file.write("%f\t" % w_i)
    
    embed_file.write("\n")

