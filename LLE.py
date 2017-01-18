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
import pickle as pkl


#Compute the similarit matrix
def compute_similarities(X):
    similarities = np.zeros((X.shape[0],X.shape[0]))
    similarities = squareform(pdist(X,'euclidean'))
    return similarities

#Get the K-NN for each element
def KNN(K,similarities):
    I = (np.argsort(similarities,1))
    I=I[:,1:K+1]
    return I



#Compute the reconstruction weights W
def compute_W(I,X,D,K):
    W=np.zeros((X.shape[0],X.shape[0]))
    for i in range(0,X.shape[0]):
        Z=X[I[i,:],:]
        Z=Z-X[i,:]
        C=np.dot(Z,np.transpose(Z))
        if K>D:
          C=C+10**(-3)*np.trace(C)*np.eye(C.shape[0])
        w=np.linalg.solve(C,np.transpose(np.ones(C.shape[0])))
        W[i,I[i,:]]=w/np.sum(w)
    return W


#Compute embedding coordinates Y using weights W
def compute_embedding(W,dim):
    M=(np.transpose((np.eye(W.shape[0])-W))).dot((np.eye(W.shape[0])-W))
    [eigenval,eigenvect]=np.linalg.eig(M)
    I=np.argsort(eigenval)[1:dim+1]
    Y=eigenvect[:,I]
    return eigenval,Y

#Compute LLE
def LLE(X,k,d):
    print("----Compute similarities----")
    sim=compute_similarities(X)
    print("----Get the neighbours----")
    I=KNN(k,sim)
    print("----Compute the reconstruction weights----")
    D=X.shape[1]
    W=compute_W(I,X,D,d)
    print("----Compute the embedding----")
    [eigenval,Y]=compute_embedding(W,d)
    return Y
