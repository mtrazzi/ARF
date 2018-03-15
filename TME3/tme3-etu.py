# AUTHOR: Michael Trazzi (mtrazzi) and Julien Denes (jdenes)
from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from arftools import plot_data, plot_frontiere, gen_arti

########################################
# Fonctions utilitaires
########################################

# Make grid
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

# Load dataset
def load_usps(filename):
    with open(filename,"r") as f:
        f.readline()
        data =[ [float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

# Plot function and gradient evolution depending on nb of iterations
def plot_nb_iter(f, df, x_init, eps, max_iter):
	x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
	plt.figure()
	plt.scatter([i for i in range(len(x_histo))], f_histo, label="f")
	plt.scatter([i for i in range(len(x_histo))], grad_histo, label="gradient")
	plt.legend()
	plt.show()	

# Plot function and optimization trajectory for f1 and f2
def plot_2d(f, df, x_init, eps, max_iter):
    x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
    r = sorted([x_init, x_histo[-1]])
    x = np.arange(r[0]-1, r[1]+1, 0.2)
    plt.plot(x, f(x), color='blue', label="f")
    plt.plot(x_histo, f_histo, 'ro', label="optimisation")
    plt.legend()
    plt.show()

# Plot function and optimization trajectory for f3
def plot_3d(f, df, x_init, eps, max_iter):
    grid,xx,yy = make_grid(-1,1,-1,1)
    plt.contourf(xx,yy,f(grid).reshape(xx.shape))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, f(grid).reshape(xx.shape),rstride=1,cstride=1,\
			cmap=cm.gist_rainbow, linewidth=0, antialiased=False)
    fig.colorbar(surf)
    x_histo, f_histo, grad_histo = optimize(f, df, x_init, eps, max_iter)
    ax.plot(x_histo[:,0], x_histo[:,1], f_histo.ravel(), color='black', label="optimisation")
    plt.legend()
    plt.show()
    
########################################
# Algorithme de descente de gradient
########################################

# Gradient descent implementation given function, gradient and other parameters
def optimize(fonc, dfonc, x_init, eps, max_iter=20):
    x_histo, f_histo, grad_histo = x_init, fonc(x_init), dfonc(x_init)
    x = x_init
    for i in range(max_iter):
        x = x - eps * dfonc(x)
        x_histo = np.vstack((x_histo, x))
        f_histo = np.vstack((f_histo, fonc(x)))
        grad_histo = np.vstack((grad_histo, dfonc(x)))
    return x_histo, f_histo, grad_histo

########################################
# Optimisation de fonctions
########################################

# A set of mathematical functions to optimize

def f1(x):
    return x * np.cos(x)

def f2(x):
    return - np.log(x) + x**2

def f3(x):
    return 100*(x[:,1] - x[:,0]**2)**2 + (1 - x[:,0])**2

def df1(x):
	return np.cos(x) - x * np.sin(x)

def df2(x):
	return -1/x + 2 * x

def df3(x):
    return np.array([200 * (-2*x[:,0]) * (x[:,1] - x[:,0]**2) + (-2) * (1 - x[:,0]),\
                            100 * (x[:,1] - x[:,0]**2)]).reshape(1,2)

########################################
# Régression logistique
########################################    
    
class Learner(object):
    
    def __init__(self, max_iter=1000, eps=0.01):
        """ :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, datax, datay):
        """ :datax: donnees de train
            :datay: label de train
        """
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))

        for i in range(self.max_iter):
            self.w = self.w - self.eps * self.grad_loss(datax, datay, self.w)
        return self.w

    def loss(self, datax, datay):
        temp = -(2*datay-1) * np.dot(datax, self.w)
        return - np.sum(np.log(1 + np.exp(temp)), axis=0)
        
    def grad_loss(self, datax, datay, w):
        temp = (2*datay-1) * np.dot(datax, w.T)
        return - np.sum(datax / (1 + np.exp(temp)), axis=0)

    def predict(self,datax):
        return np.sign(1/(1 + np.exp(- np.dot(datax, self.w.T))) - 0.5).reshape(-1)

    def score(self,datax,datay):
        return np.mean(self.predict(datax) != datay)
    
########################################
# Main
########################################

def main():
    """
    # Affichage de f et df en fonction du nombre d'itérations
    plot_nb_iter(f1, df1, 5, 0.1, 20)
    plot_nb_iter(f2, df2, 5, 0.1, 20)
    
    # Affichage 2D de f1, f2 et 3D de f3
    plot_2d(f1, df1, 5, 0.1, 20)
    plot_2d(f2, df2, 5, 0.1, 20)
    plot_3d(f3, df3, np.array([-1,0]).reshape(1,2), 0.001, 20)
    """
    # Regression linéaire avec données USPS
    datax_train, datay_train = load_usps("USPS_test.txt")
    datax_test, datay_test = load_usps("USPS_train.txt")
    
    # On test la reconnaissance 6 vs 9 (on isole puis transforme les données en -1 et 1)
    datax_train_2 = datax_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    datay_train_2 = datay_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    labely_train = np.sign(datay_train_2 - 2)
    datax_test_2 = datax_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    datay_test_2 = datay_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    labely_test = np.sign(datay_test_2 - 2)

    # Construction et entrainement
    model = Learner(max_iter = 100, eps = 0.5)
    model.fit(datax_train_2, labely_train)
    print("Erreur de classification 6/9: train %f, test %f"\
          % (model.score(datax_train_2,labely_train),model.score(datax_test_2,labely_test)))

    # Essai classification 1 versus toutes les autres
    model2 = Learner(max_iter = 100, eps = 0.5)
    labely_train = 2 * (datay_train == 6) - 1
    labely_test = 2 * (datay_test == 6) - 1
    model2.fit(datax_train, labely_train)
    print("Erreur one vs all: train %f, test %f"\
          % (model2.score(datax_train,labely_train),model2.score(datax_test,labely_test)))

    # Essai aussi avec gen_arti pour tests de performances
    trainx, trainy = gen_arti(nbex=1000, data_type=0,epsilon=1)
    testx, testy = gen_arti(nbex=1000, data_type=0,epsilon=1)
    model1 = Learner(max_iter = 100, eps = 0.01)
    model1.fit(trainx,trainy)
    print("Erreur de classification gen_arti: train %f, test %f"\
          % (model1.score(trainx,trainy),model1.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx, model1.predict, 200)
    plot_data(trainx, trainy)



if __name__ == "__main__":
    main()