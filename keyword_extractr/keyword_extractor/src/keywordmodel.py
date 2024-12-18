import joblib
import pandas as pd
import re
import contractions
import seaborn as sns
import spacy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import umap
import nltk
import numpy as np

from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from collections import defaultdict
from spacy.cli import download

# download("en_core_web_sm")
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
class KeywordExtractionTesting():
  def __init__(self,kmeanmod,tfidfmodel,umap_model):
    """
    Initialize and load models
    """
    self.kmeans=joblib.load(kmeanmod)
    print("Kmeans loaded...")
    self.tfidf=joblib.load(tfidfmodel)
    print("TFIDF loaded...")
    self.nlp=spacy.load('en_core_web_sm')
    self.model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Sentence Transformer loaded...")
    self.umap=joblib.load(umap_model)
    self.ngram_feature=self.tfidf.get_feature_names_out()
  def clean_text(self,text):
    """
    Preprocessing the data

    """
    # Remove numbers, .com, and hyperlinks
    text= contractions.fix(text)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b\w+\.com\b', '', text)  # Remove .com URLs
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove hyperlinks
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    # words = nltk.word_tokenize(text)
    # pos_tags = pos_tag(words)
    # processed_words = [
    #     lemmatizer.lemmatize(word, get_wordnet_pos(tag)).lower()
    #     for word, tag in pos_tags
    #     if word.lower() not in stop_words  # Remove stopwords
    # ]
    doc=self.nlp(text)
    filtered_words = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(filtered_words)
  def get_top_n(self,all_keywords,labels,topk=10):
    """
    Get topk keywords
    """
    topk=topk
    cluster_imp_key={}
    for data,lbl in zip(all_keywords,job_labels):
      keywords=sorted(data.items(), key=lambda x: x[1], reverse=True)[:topk]
      cluster_imp_key[lbl]=keywords
    return cluster_imp_key
  def process(self,text):
    """
    Process and extract he keywordfrom the text
    """
    resultant_ngram={}
    processed_text=self.clean_text(text)
    embed=self.model.encode([processed_text])
    embed_comp=self.umap.transform(embed.reshape(1,-1))
    label=self.kmeans.predict(embed_comp)
    tfidffeature=self.tfidf.transform([processed_text])
    row=tfidffeature[0]
    indices=row.nonzero()
    feature_words=[]
    for idx in indices:
      feature_words.append(self.ngram_feature[idx])
    for ngramword in feature_words[0]:
      embed_word=self.umap.transform(self.model.encode([ngramword]).reshape(1,-1))
      sim=cosine_similarity(embed_comp.reshape(1,-1),embed_word)
      resultant_ngram[ngramword]=sim[0][0]
    
    impkeys=self.get_top_n(resultant_ngram,label)
    return impkeys






    