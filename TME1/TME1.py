# -*- coding: latin-1 -*-
# CODING : PYTHON 3.4
# AUTHOR: Michaël TRAZZI et Julien DENES
"""
Created on Wed Feb  7 10:46:41 2018

@author: 3770924
"""
from __future__ import division
import pickle
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

# data : tableau ( films , features ) , id2titles : dictionnaire id -> titre ,
# fields : id feature -> nom
[ data , id2titles , fields ] = pickle.load(open("imdb_extrait.pkl","rb"))
# la derniere colonne est le vote
datax = data [: ,:32]
datay = np . array ([1 if x [33] >6.5 else -1 for x in data ])

def entropie(vect):
    _, counts = np.unique(vect, return_counts=True)
    p_y = np.array(counts / len(vect))
    return (-np.sum(p_y * np.log(p_y)))
    
#test = [1, 7, 1, 3, 3, 3, 3, 3, 9, 9]
#bi_test = [[1, 7, 1, 3, 3, 3, 3, 3, 9, 9], [3, 3, 3, 3, 3, 9, 9]]     
#entropie(test)

def entropie_cond(list_vect):
    n = len(list_vect)
    total_nb = sum(len(part) for part in list_vect)
    p = np.array((1,n))
    H = np.array((1, n))
    for i in range(n):
            p[i] = len(list_vect[i]) / total_nb
            H[i] = entropie(list_vect[i])
    return (np.sum(H * p))

#TO DO : Q1.3.



from decisiontree import DecisionTree
def predict(depth, x, y ,x_test, y_test):
    dt = DecisionTree()
    dt.max_depth = depth # on fixe la taille de l ’ arbre a 5
    dt.min_samples_split = 2 # nombre minimum d ’ exemples pour spliter un noeud
    dt.fit ( x , y )
    dt.predict ( x_test [:5 ,:])
    score = dt.score ( x_test , y_test )
    print(score)
    return(score)

# dessine l ’ arbre dans un fichier pdf si pydot est installe .
#dt.to_pdf ("/ tmp / test_tree . pdf ", fields )
# sinon utiliser http :// www . webgraphviz . com /
#dt . to_dot ( fields )
# ou dans la console
#print (dt.to_dot( fields ))

## Q1.4: Plus on utilise des arbres profonds, et plus on surapprend notre base de données d'apprentissage
## C'est classique

## Q1.5: Le score est : 0.900152605189 pour une profondeur de 50
## Pour profondeur de 5: 0.736429038587

## Q1.6: Le score n'est pas un indicateur fiable. Il faut utiliser des cross-validation dataset et test set
## On surapprend sur le dataset d'apprentissage

def train(datax, datay):
    n = len(datax)
    score_mat = [[0]*10 for i in range(3)]
    for fraction_learning in [0.2, 0.5, 0.8]:
        print('fraction_learning is ',fraction_learning)
        indexes = [i for i in range(n)]
        shuffle(indexes)
        training_set_indexes = indexes[:int(fraction_learning*n)]
        test_set_indexes = indexes[int(fraction_learning*n):]
        #print(training_set_indexes)
        learning_set_x, learning_set_y = np.array([datax[j] for j in training_set_indexes]), np.array([datay[j] for j in training_set_indexes])
        test_set_x, test_set_y = np.array([datax[j] for j in test_set_indexes]), np.array([datay[j] for j in test_set_indexes])
        #print(learning_set_x, learning_set_y)    
        #predict(5, learning_set_x, learning_set_y)
        for i in range(10):
            index = [0.2, 0.5, 0.8].index(fraction_learning)
            score_mat[index][i] = predict(5*(i+1), learning_set_x, learning_set_y, test_set_x, test_set_y)
        print()
    plt.plot(score_mat[0], label="(0.2,0.8)")
    plt.plot(score_mat[1], label="(0.5,0.5)")
    plt.plot(score_mat[2], label="(0.8,0.2)")
    plt.legend()
    plt.show()
    return score_mat

# Q1.7: pour une profondeur de 5 0.70765206017 0.733594942228 0.739045127534

def main():
    global datax, datay;
    score_mat = train(datax, datay)
    print(*score_mat, sep='\n')
    return 0

if  __name__ == "__main__":
    main()
