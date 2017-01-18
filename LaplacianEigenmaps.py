# -*- coding: utf-8 -*-
"""

@author: Trichelair Paul
"""
import numpy as np
import matplotlib.pyplot as plt;
from scipy.spatial.distance import *
from scipy.sparse.linalg import eigs
import pandas as pd
import csv
import pickle


class Graph():
    #Graph is dedicated to compute the adjacency graph, the weights and the Laplacian
    def __init__(self,graph_type,graph_thresh,t):
        self.graph_type=graph_type
        self.graph_thresh=graph_thresh
        self.t=t
        self.W=0
        self.D=0
        self.L=0
        self.similarities=0
    def compute_similarities(self,X):
        similarities = np.zeros((X.shape[0],X.shape[0]))
        similarities = squareform(pdist(X,'euclidean'))
        a = np.exp(np.divide(-similarities,float(self.t)))
        self.similarities=a
    def fit(self,X):
        self.compute_similarities(X)
        self.W=np.zeros((X.shape[0],X.shape[0]))
        self.W=self.similarities
        if self.graph_type=='eps':
            self.W=np.dot(self.W,(self.W>=self.graph_thresh))
        else:
            I = (np.argsort(self.similarities,1))
            I=I[:,-self.graph_thresh:];
        M=np.zeros(self.W.shape)
        for i in range(0,X.shape[0]):
            M[i,I[i,:]]=self.W[i,I[i,:]]
        self.W=np.maximum(M,np.transpose(M))
        self.D=np.zeros(self.W.shape)
        self.D[np.diag_indices_from(self.D)]=np.sum(self.W,axis=1)
        self.L=self.D-self.W
    def get_weights(self):
        return self.W
    def get_D(self):
        return self.D
    def get_Laplacian(self):
        return self.L


class SpectralEmbedding():
    #Implement the embedding
    def __init__(self,graph,dim):
        self.dim=dim
        self.graph=graph
        self.f=0
    def embedding(self):
        L_rw=np.dot(np.linalg.inv(self.graph.get_D()),self.graph.get_Laplacian())
        [eigenval,eigenvect]=np.linalg.eig(L_rw)
        I=np.argsort(eigenval)[1:self.dim+1]
        self.f=eigenvect[:,I]
        return self.f
