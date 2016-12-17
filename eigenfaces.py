#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 23:09:33 2016

@author: quien
"""

import glob;

import numpy as np;
import numpy.random as rd;

import matplotlib.image as img;
import matplotlib.pyplot as plt;

names = glob.glob("att_faces/*/*.pgm");

A = None;
s = 4;
for name in names:
    a = img.imread(name)/255.0;
    b = np.zeros((a.shape[0]/s,a.shape[1]/s));
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b[i,j] = np.mean(a[s*i:s*i+s,s*j:s*j+s]);
    print name,b.shape;
    if A is None:
        A = b.reshape((b.shape[0],b.shape[1],1));
    else:
        A = np.concatenate((A,b.reshape((b.shape[0],b.shape[1],1))),axis=2);
    
mu_A = np.mean(A,axis=2);

plt.imshow(mu_A);

A_cen = A-np.repeat(mu_A.reshape((mu_A.shape[0],mu_A.shape[1],1)),A.shape[2],axis=2);

sig_A = np.zeros((A.shape[0]*A.shape[1],A.shape[0]*A.shape[1]));

for i in range(A.shape[2]):
    aux_A = A[:,:,i].reshape((A.shape[0]*A.shape[1],1));
    sig_A += np.outer(aux_A,aux_A);

sig_A /= A.shape[2];
sig_A += 1e-3*np.eye(A.shape[0]*A.shape[1]);

plt.imshow(sig_A);

E,V = np.linalg.eig(sig_A);

V = np.real(V.reshape((A_cen.shape[0],A_cen.shape[1],V.shape[1])));

E = np.real(E);
E = np.sqrt(E);

B = np.dot(V,np.diag(E));

def gen():
    return np.dot(B,rd.randn(V.shape[2]))+mu_A;
    
r = gen()
plt.imshow(r)
