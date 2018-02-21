# AUTHOR: Michael Trazzi (mtrazzi) and Julien Denes (jdenes)
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def make_grid(xmin=-5,xmax=5,ymin=-5,ymax=5,step=20,data=None):
    """ Cree une grille sous forme de matrice 2d de la liste des points
    :return: une matrice 2d contenant les points de la grille, la liste x, la liste y
    """
    if data is not None:
        xmax,xmin,ymax,ymin = np.max(data[:,0]),np.min(data[:,0]),\
                              np.max(data[:,1]),np.min(data[:,1])
    x,y = np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step),
                      np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

########################################

# GRADIENT DESCENT

########################################

def optimize(fonc, dfonc, x_init, eps, max_iter):
    x_histo, f_histo, grad_histo = [x_init], [fonc(x_init)], [dfonc(x_init)]
    x = x_init
    for i in range(max_iter):
        x = x - eps * dfonc(x)
        x_histo.append(x)
        f_histo.append(fonc(x))
        grad_histo.append(dfonc(x))
    return np.array(x_histo), np.array(f_histo), np.array(grad_histo)

########################################

# OPTIMISATION DE FONCTIONS

########################################

from math import cos,log

def f1(x):
    return x * cos(x)

def f2(x):
    return -log(x) + x**2

def f3(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def print2d(myfunction):
    grid,xx,yy = make_grid(-1,3,-1,3,20)
    plt.figure()
    plt.contourf(xx,yy,myfunction(grid).reshape(xx.shape))
    
    
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, myfunction(grid).reshape(xx.shape),rstride=1,cstride=1,\
    	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()

#print2d(f1)

def grad_app(f,eps):
    return lambda x : ((f(x + eps) - f(x - eps)) / (2 * eps))

print(optimize(f1, grad_app(f1, .001), 0, .005, 50))
print(optimize(f2, grad_app(f1, .001), 5, .005, 50))
print(optimize(f3, grad_app(f1, .001), np.array([8, 12]), .005, 50))

