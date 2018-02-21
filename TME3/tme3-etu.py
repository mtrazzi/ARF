# AUTHOR: Michael Trazzi (mtrazzi) and Julien Denes (jdenes)
from __future__ import print_function
from __future__ import division
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
        x = np.add(x,-eps * dfonc(x))
        x_histo.append(x)
        f_histo.append(fonc(x))
        grad_histo.append(dfonc(x))
    return x_histo, f_histo, grad_histo

########################################

# OPTIMISATION DE FONCTIONS

########################################

from math import cos,log,sin

def f1(x):
    return x * np.cos(x)

def f2(x):https://github.com/mtrazzi/ARF.git
    return -log(x) + x**2

def f3(x):
	return 100*(x[:,1] - x[:,0]**2)**2 + (1 - x[:,0])**2

def df1(x):
	return cos(x) - x * sin(x)

def df2(x):
	return -1/x + 2 * x

def df3(x):
	if (len(x.shape == 1)):
		x = np.expand_dims(x, axis=0)
	return np.array([200 * (-2*x[:,0]) * (x[:,1] - x[:,0]**2) + (-2) * (1 - x[:,0]), 100 * (x[:,1] - x[:,0]**2)])

def print2d(myfunction):
    grid,xx,yy = make_grid(-1,3,-1,3,20)
	#print(np.shape(grid))
    plt.figure()
    plt.contourf(xx,yy,myfunction(grid).reshape(xx.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, myfunction(grid).reshape(xx.shape),rstride=1,cstride=1,\
    	  cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()

def grad_app(f,eps):
    return lambda x : ((f(x + eps) - f(x - eps)) / (2 * eps))

######

# TEST

######
#print(optimize(f1, df1, 0, .005, 50))
#print(optimize(f2, df2, 5, .005, 50))
#print(optimize(f3, df3, np.array([8, 12]), .005, 50))

#print2d(f3)

def print_nb_iter(f, df, x_init, eps, max_iter):
	x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
	plt.figure()
	plt.scatter([i for i in range(len(x_histo))], f_histo, label="f")
	plt.scatter([i for i in range(len(x_histo))], grad_histo, label="gradf")
	plt.legend()
	plt.show()	


#print_nb_iter(f1, df1, 0, 0.1, 100)
#print_nb_iter(f2, df2, 5, 0.1, 100)

def print3d(f, df, x_init, eps, max_iter):
	fig = plt.figure()
	x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
	ax = fig.add_subplot(111, projection='3d')
	ax.plot(x_histo[:,0],x_histo[:,1], f_histo.ravel())
	ax.plot(x_histo[:,0],x_histo[:,1], grad_histo.ravel())
	plt.legend()
	plt.show()

def print3d_bis(f, df, x_init, eps, max_iter):
    grid,xx,yy = make_grid(-1,1,-1,1)
	#print(np.shape(grid))
    plt.contourf(xx,yy,f(grid).reshape(xx.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, f(grid).reshape(xx.shape),rstride=1,cstride=1,\
			cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
    ax.plot(x_histo[:,0], x_histo[:,1], f_histo.ravel(), color='black')
    plt.show()
###
# RESIGN 3D
###

print3d_bis(f3, df3, np.array([[3,5]]).reshape(2,1), 0.1, 100)


def print_f_and_approx(f, df, x_init, eps, max_iter):
	x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
	plt.figure()
	plt.scatter(x_histo, f_histo, label="f", color='red')
	plt.plot(x_histo, [f(x) for x in x_histo], label="f(x_i)")
	plt.legend()
	plt.show()	

print_f_and_approx(f1, df1, 0, 0.1, 100)
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

