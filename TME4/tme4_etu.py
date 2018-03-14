from arftools import plot_data, plot_frontiere, make_grid, gen_arti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc

@decorator_vec
def mse(datax,datay,w):

    pred = np.dot(datax, w.T)
    res = np.sum((pred - datay)**2)
    return (1/np.shape(datax)[0]) * res

@decorator_vec
def mse_g(datax,datay,w):

    M = datay - np.dot(datax, w.T)
    return (-2/np.shape(datax)[0]) * np.sum(np.dot(datax.T, M))

@decorator_vec
def hinge(datax,datay,w):

    return (1/np.shape(datax)[0]) * np.sum(np.maximum(0, -datay*np.dot(datax, w.T)))

@decorator_vec
def hinge_g(datax,datay,w):

   sign = np.sign(hinge(datax, datay, w))
   res = - sign * datax * datay
   return (1/np.shape(datax)[0]) * np.sum(res, axis=0)


class Lineaire(object):
    def __init__(self,loss=hinge,loss_g=hinge_g,max_iter=1000,eps=0.01):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter,eps
        self.loss, self.loss_g = loss, loss_g
        self.bias = False
        self.project = None
        self.base = None

    # A general function to add projections
    def projection(self,datax):

        res = datax
        
        if (self.project == "gauss"):
            res = np.zeros([datax.shape[0], self.base.shape[0]])
            for i, x in enumerate(datax):
                for j, b in enumerate(self.base):
                    res[i,j] = np.exp(-np.linalg.norm(x-b)**2/0.1)

        if(self.bias):
            vector = np.ones((res.shape[0],1))
            res = np.column_stack([vector, res])

        return res

    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        datax = self.projection(datax)
        datay = datay.reshape(-1,1)
        N = len(datay)
        datax = datax.reshape(N,-1)
        D = datax.shape[1]
        self.w = np.random.random((1,D))

        i = 0
        for i in range(self.max_iter):
            self.w = self.w - self.eps * self.loss_g(datax, datay, self.w)
        return self.w

    def predict(self,datax):
        datax = self.projection(datax)
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.sign(np.dot(datax, self.w.T)).reshape(-1)

    def score(self,datax,datay):
        return np.mean(self.predict(datax) != datay)

# Utility functions
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def show_usps(data):
    plt.figure()
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()
    plt.show()

def plot_error(datax,datay,f,step=10):
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()



if __name__=="__main__":

    """ Tracer des isocourbes de l'erreur """
    plt.ion()
    trainx,trainy = gen_arti(nbex=1000,data_type=0,epsilon=1)
    testx,testy = gen_arti(nbex=1000,data_type=0,epsilon=1)
    base = trainx[np.random.randint(trainx.shape[0], size=100), :] #base pour proj gaussienne
#    plt.figure()
#    plot_error(trainx,trainy,mse)
#    plt.figure()
#    plot_error(trainx,trainy,hinge)
    
#    # MSE
#    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1)
#    perceptron.fit(trainx,trainy)
#    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
#    plt.figure()
#    plot_frontiere(trainx,perceptron.predict,200)
#    plot_data(trainx,trainy)
    
#    # Hinge
#    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
#    perceptron.fit(trainx,trainy)
#    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
#    plt.figure()
#    plot_frontiere(trainx,perceptron.predict,200)
#    plot_data(trainx,trainy)
    
#    # Add bias
#    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.1)
#    perceptron.bias = True
#    perceptron.fit(trainx,trainy)
#    print("Erreur : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
#    plt.figure()
#    plot_frontiere(trainx,perceptron.predict,200)
#    plot_data(trainx,trainy)

    # Hinge with gaussian transformation
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.bias = False
    perceptron.project = "gauss"
    perceptron.base = base
    perceptron.fit(trainx,trainy)
    print("Erreur transformation : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)

"""
    datay_train
    # Données USPO
    datax_train, datay_train = load_usps("USPS_test.txt")
    datax_test, datay_test = load_usps("USPS_train.txt")
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(datax_train,datay_train)
    print("Erreur : train %f, test %f"% (perceptron.score(datax_train,datay_train),perceptron.score(datax_test,datay_test)))

    #6 vs 9
    two_class_datax_train = datax_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    two_class_datay_train = datay_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    labely_train = np.sign(two_class_datay_train - 7)
    two_class_datax_test = datax_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    two_class_datay_test = datay_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    labely_test = np.sign(two_class_datay_test - 7)


    perceptron.fit(two_class_datax_train, labely_train)
    print("Erreur 2 classes 6/9: train %f, test %f"% (perceptron.score(two_class_datax_train,labely_train),perceptron.score(two_class_datax_test,labely_test)))

    #1 vs 2
    two_class_datax_train = datax_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    two_class_datay_train = datay_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    labely_train = np.sign(two_class_datay_train - 2)
    two_class_datax_test = datax_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    two_class_datay_test = datay_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    labely_test = np.sign(two_class_datay_test - 2)


    perceptron.fit(two_class_datax_train, labely_train)
    print("Erreur 2 classes 1/8: train %f, test %f"% (perceptron.score(two_class_datax_train,labely_train),perceptron.score(two_class_datax_test,labely_test)))

    #Uncomment to Plot the weights
    #print(*perceptron.w, sep='\n')
    #6 vs all
    labely_train = 2 * (datay_train == 6) - 1
    labely_test = 2 * (datay_test == 6) - 1
    perceptron.fit(datax_train, labely_train)
    print("Erreur one vs all: train %f, test %f"% (perceptron.score(datax_train,labely_train),perceptron.score(datax_test,labely_test)))
"""
