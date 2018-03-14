#Coding Python 3.5
#AUTHOR : Michael Trazzi and Julien Denes
from sklearn import svm
import pickle
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from arftools import *
import hinge

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data=[[float(x)for x in l.split()]for l in f if len(l.split()) >2 ]
        tmp=np.array(data)
        returntmp[:,1:],tmp[:,0].astype(int)

def plot_frontiere_proba(data,f,step):
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),255)

def fit_kernel(datax, datay, kernel):
    clf = svm.SVC(probability=True, kernel=kernel)
    clf.fit(datax, datay)
    return clf

def main():

    #################

    # SVM/GRID SEARCH

    #################

    trainx, trainy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    testx,testy = gen_arti(nbex=1000,data_type=0,epsilon=1)
    clf = svm.SVC(probability=True)
    clf.fit(trainx, trainy)
    #for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    #    clf = fit_kernel(trainx, trainy, kernel)
    #    plot_frontiere_proba(testx,lambda x : clf.predict_proba(x)[:,0],step=50)
    #    plot_data(testx, testy)
    #    plt.show()

    # Testing gaussian parameters

    #for sigma_2 in np.linspace(0.001, 1, 100):
    #    clf = svm.SVC(probability=True, kernel='rbf')
    #    clf.fit(datax, datay)


    # GRID SEARCH
    C_2d_range = [1e-2, 1, 1e2]
    gamma_2d_range = [1e-1, 1, 1e1]
    classifiers = []
    for C in C_2d_range:
        for gamma in gamma_2d_range:
            clf = SVC(C=C, gamma=gamma)
            clf.fit(X_2d, y_2d)
            classifiers.append((C, gamma, clf))

    ## NOTE : Parametre C depend du nombre d'exemples. Utiliser NuSVC pour qqchose d'indépendant du nb d'exemples

    #print(clf.support_vectors_) #Support vectors : plus il y a de support vector, plus on a un modele complexe et on surapprend

    ## Noyau Polynomial : cf ce qui avait ete fait TME precedent.
    # Projection polynomiale : K(x,y) = (1 + <x,y>)^2 = < phi(x), phi(y) >

    ############################

    # APPRENTISSAGE MULTI-CLASSE

    # SVM ET GRID-SEARCH

    ############################

    # Not to do


    return 0

if __name__=="__main__":
    main()