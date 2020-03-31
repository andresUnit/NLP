# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 15:28:04 2020

@author: AMS
"""
__version__ = 0.1
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
import warnings


warnings.filterwarnings("ignore")

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

### -------- llevar texto a espacio semantico -------- ###

def encoder(text):
    embeddings = model.encode(text)
    return embeddings

### -------- Calcular distancia semantica, calcula el angulo entre vectores -------- ###

def dist(text1,text2):
    aux1 = encoder([text1])
    aux2 = encoder([text2])
    distances = distance.cdist(aux1, aux2, "cosine")[0]
    return distances

### -------- asignacion de notas segun distancia semantica -------- ###

def calificar(text1,text2, distance = False):
    if (pd.isnull(text1) | (pd.isnull(text2))):
        return 0.0
    distancia = semejanza(text1,text2)
    if distance:
        return distancia[0]
    if distancia[0]< .25:
        return 1.0
    elif distancia[0]<.43:
        return 2.0
    elif distancia[0]<.52:
        return 3.0
    elif distancia[0]<.60:
        return 4.0
    elif distancia[0]<.68:
        return 5.0
    elif distancia[0]<.75:
        return 6.0
    else:
        return 7.0
    
### -------- Semejanza entre cadenas, usa la distancia semantica valores [0,1] (mas alto mejor) -------- ###
    
def semejanza(text1,text2):    
    if (pd.isnull(text1) | (pd.isnull(text2))):
        return 0.0
    distances = dist(text1,text2)
    return 1 - distances

### -------- Cluster de texto, recibe una lista de cadenas, devuelve una lista de listas de cadenas -------- ###

def clusterizar(data, model = 'kmeans', ncluster = 5, optimizar = True, metodo_opt = 'wss', n_k = 10 ,show = True):
    data = data[(~pd.isnull(data)) & (data != '.') ]
    data = data.tolist()
    data_embeddings = encoder(data)
    dicMetod = {'wss':'Sum of Squared Distances', 'sil':'Silhouette Method'}
    if model == 'kmeans':
        if optimizar:
            if n_k != 10:
                if metodo_opt == 'wss':
                    vecOPT = calculate_WSS(data_embeddings,kmax = n_k)
                elif metodo_opt == 'sil':
                    vecOPT = calculate_SIL(data_embeddings,kmax = n_k)
            else:
                if metodo_opt == 'wss':
                    vecOPT = calculate_WSS(data_embeddings)
                elif metodo_opt == 'sil':
                    vecOPT = calculate_SIL(data_embeddings)
            if show:
                plt.plot([ i for i in range(2,len(vecOPT)+2)], vecOPT, 'bx-')
                plt.xlabel('k')
                plt.ylabel(dicMetod[metodo_opt])
                plt.title('{} para k optimo'.format(dicMetod[metodo_opt]))
                plt.grid()
                plt.show()
            ncluster = optimo(vecOPT, metodo_opt)
        clustering_model = KMeans(n_clusters = ncluster)
        clustering_model.fit(data_embeddings)
        cluster_assignment = clustering_model.labels_
        centroides = clustering_model.cluster_centers_
        clustered_sentences = [[] for i in range(ncluster)]
        for sentence_id, cluster_id in tqdm(enumerate(cluster_assignment)):
            clustered_sentences[cluster_id].append(data[sentence_id])
        return clustered_sentences, centroides
    elif model == 'affinity':
        clustering_model = AffinityPropagation(max_iter = 100000000)
        clustering_model.fit(data_embeddings)
        cluster_assignment = clustering_model.labels_
        centroides = clustering_model.cluster_centers_
        clustered_sentences = [[] for i in range(len(clustering_model.cluster_centers_indices_))]
        for sentence_id, cluster_id in tqdm(enumerate(cluster_assignment)):
            clustered_sentences[cluster_id].append(data[sentence_id])
        return clustered_sentences, centroides

### -------- Cluster de texto, Funciones de optimización Kmeans -------- ###

def calculate_WSS(points, kmax = 10):
  WSS = []
  for k in range(2, kmax + 1):
    kmeans = KMeans(n_clusters = k, random_state = 42).fit(points)
    WSS.append(kmeans.inertia_)
  return WSS

def calculate_SIL(points, kmax = 8):
    SIL = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters = k, random_state = 42).fit(points)
        labels = kmeans.labels_
        SIL.append(silhouette_score(points, labels, metric = 'euclidean'))
    return SIL

def optimo (vecOPT, metodo):
    if metodo == 'wss':
        serie = pd.Series(vecOPT)
        return serie.idxmin() + 2
    elif metodo == 'sil':
        serie = pd.Series(vecOPT)
        return serie.idxmax() + 2

def parameter_cluster(clustered_sentences,centroides):
    parameter = pd.DataFrame(index = ['count','mean','std','min','25%','50%','75%','max'])
    representative_sentences = []
    for n_cluster in range(len(clustered_sentences)):
        centroide = centroides[n_cluster]
        suma = []
        for n_comentario in range(len(clustered_sentences[n_cluster])):
            frase = [clustered_sentences[n_cluster][n_comentario]]
            embedding =  encoder(frase)
            distancia = distance.cdist([centroide],embedding, 'cosine')[0]
            suma.append(1-distancia[0])
        aux = pd.Series(suma)
        ind_max = aux.idxmax()
        main_sentence = clustered_sentences[n_cluster][ind_max]
        representative_sentences.append(main_sentence)
        parameter['Cluster {}'.format(n_cluster + 1)] = aux.describe()
    return parameter, representative_sentences
