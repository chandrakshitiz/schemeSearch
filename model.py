import os
import pandas as pd
import numpy as np
import pickle
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
from gensim.models import FastText #f
from rank_bm25 import BM25Okapi
import nmslib
import time
import pandas as pd

df = pd.read_csv('mycsv.csv', index_col=0)
nlp = spacy.load("en_core_web_sm")
tok_text=[] # for our tokenised corpus
text = df.desc.str.lower().values

for doc in tqdm(nlp.pipe(text, disable=["tagger", "parser","ner"])):
    tok = [t.text for t in doc if (t.is_ascii and not t.is_punct and not t.is_space)]
    tok_text.append(tok)

ft_model = FastText(
    sg=1,
    vector_size=100, 
    window=10, 
    min_count=5, 
    negative=15, 
    min_n=2,
    max_n=5 
)

ft_model.build_vocab(tok_text)

ft_model.train(
    tok_text,
    epochs=6,
    total_examples=ft_model.corpus_count, 
    total_words=ft_model.corpus_total_words)

ft_model.save('_fasttext.model')
    
ft_model = FastText.load('_fasttext.model')

bm25 = BM25Okapi(tok_text)
weighted_doc_vects = []

for i,doc in tqdm(enumerate(tok_text)):
  doc_vector = []
  for word in doc:
      vector = ft_model.wv[word]
      weight = (bm25.idf[word] * ((bm25.k1 + 1.0)*bm25.doc_freqs[i][word])) 
      (bm25.k1 * (1.0 - bm25.b + bm25.b *(bm25.doc_len[i]/bm25.avgdl))+bm25.doc_freqs[i][word])
      weighted_vector = vector * weight
      doc_vector.append(weighted_vector)
  doc_vector_mean = np.mean(doc_vector,axis=0)
  weighted_doc_vects.append(doc_vector_mean)

pickle.dump(weighted_doc_vects, open( "weighted_doc_vects.p", "wb" ) )

with open( "weighted_doc_vects.p", "rb" ) as f:
  weighted_doc_vects = pickle.load(f)
# create a random matrix to index
data = np.vstack(weighted_doc_vects)

# initialize a new index, using a HNSW index on Cosine Similarity - can take a couple of mins
index = nmslib.init(method='hnsw', space='cosinesimil')
index.addDataPointBatch(data)
index.createIndex({'post': 2}, print_progress=True)

def recommendations(search):
   # querying the index:
#search = 'Himachel pradesh government plastic buy back scheme'
    input = search.lower().split()


    query = [ft_model.wv[vec] for vec in input]
    query = np.mean(query,axis=0)

    t0 = time.time()
    ids, distances = index.knnQuery(query, k=10)
    t1 = time.time()
    print("Query : ", search)
    print("="*100)
    print(f'Searched {df.shape[0]} records in {round(t1-t0,4) } seconds \n')
    l=[]
    for i,j in zip(ids,distances):
        print(round(j,2))
        l.append(df.desc.values[i])
        print("="*100)
    return l