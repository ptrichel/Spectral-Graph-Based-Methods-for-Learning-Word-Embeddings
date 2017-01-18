# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 11:53:40 2016

@author: Souhail
"""

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import shortest_path
import matplotlib.pylab as plt
import cPickle as pkl
from scipy.sparse.csgraph import dijkstra


def similarity_function(X, sigma):
    N = len(X)
    dis = np.zeros((N,N))
    for i in range(N):
        dis[i,i] = 0
        for j in range(i+1,N):
            dis[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            dis[j,i] = dis[i,j]
    return dis

def build_similarity_graph(graph_parameters,similarities):
    grf_type = graph_parameters[0]
    grf_thresh = graph_parameters[1]

    N = len(similarities)
    W = np.zeros((N,N));

    if grf_type == 'knn':
        tmp = np.ones((N,N))
        for i in range(N):
            ind = np.argsort(similarities[i,:])
            tmp[i,ind[grf_thresh+1:N]] = 0
        tmp = tmp + tmp.transpose()
        tmp = tmp <> 0
        W = np.multiply(similarities,tmp)
        
    elif grf_type == 'eps':
        W = similarities
        W[W < grf_thresh] = 0
    else:
        print('Cannot recognize the type of graph')
    return W

def step_3(D,d):
    N = len(D)
    H = np.identity(N)-float(1)/N*np.ones((N,N))
    tau_D = -0.5*H.dot(np.multiply(D,D).dot(H))
    [eigval,eigvec] = np.linalg.eig(tau_D)
    sqrt_eigval = np.sqrt(eigval)
    res = eigvec.dot(np.diag(sqrt_eigval))
    return res[:,0:d]
    
def isomap(graph_parameters,d, data = None, similarities = None, normalize_data = False):
    if data is None:
        if similarities is None:
            print ("Isomap needs data")
    else:
        if normalize_data:
            print("----Normalization step----")
            data_norm = normalize(data, axis = 1)
        data_norm = data
        similarities = similarity_function(data_norm,1)
    print("----Building graph----")
    W = build_similarity_graph(graph_parameters,similarities)
    print("----Computing short paths----")
    D = dijkstra(W, directed = False)
    print("----Computing embeddings----")
    res = step_3(D,d)
    return res



# Embeddings
data_path = "C:/Users/Souhail/Desktop/Master MVA/Graphs in ML/project/data/cooccurrence_matrix"
data_brut = np.loadtxt(open("".join([data_path,"/count_matrix_whs=2_dirty=True.txt"]),'rb'))
word_occur = pkl.load(open("".join([data_path,'/words_occurences.p']),'rb'))

# Normalize data
N = len(data_brut)
for i in range(N):
    print i
    for j in range(i,N):
        data_brut[i,j] = data_brut[i,j]/(np.sqrt(word_occur[i]*word_occur[j]))
        
# PPMI normalization
N = len(data_brut)
T = sum(data_brut)
TT=sum(T)


for i in range(N):
    print i
    for j in range(i,N):
        data_brut[i,j] = max(0,np.log(TT*data_brut[i,j]/(T[i]*T[j])))

# TF IDF normalization
N = len(data_brut)
T = sum(data_brut<>0)

for i in range(N):
    print i
    for j in range(N):
        data_brut[i,j] = data_brut[i,j]*np.log(float(N)/T[j])
    data_brut[i,:] = data_brut[i,:]/np.linalg.norm(data_brut[i,:])
        

data_norm = np.triu(data_brut, k =0) + np.triu(data_brut, k =1).transpose()
        
similarities = similarity_function(data_norm,1)
#pkl.dump(similarities,open('sim.p','wb'))

similarities = pkl.load(open('sim.p','rb'))

res = isomap(['knn',20,1],50, similarities = similarities)


    
    
    