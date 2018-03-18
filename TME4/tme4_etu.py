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

########################################
# Implémentation
########################################

# Moindres carrés
@decorator_vec
def mse(datax,datay,w):
    pred = np.dot(datax, w.T)
    return np.mean((pred-datay)**2)

# Gradient moindres carrés
def mse_g(datax,datay,w):
    M = datay - np.dot(datax, w.T)
    return (-2/np.shape(datax)[0]) * np.sum(np.dot(datax.T, M))

# Hinge
@decorator_vec
def hinge(datax,datay,w):
    return np.mean(np.maximum(0, -datay*np.dot(datax, w.T)))

# Gradient hinge
@decorator_vec
def hinge_g(datax,datay,w):
   sign = np.sign(hinge(datax, datay, w))
   return np.mean(( -sign * datax * datay), axis=0)

# Classe d'apprentissage
class Lineaire(object):
    
    def __init__(self, loss=hinge, loss_g=hinge_g, max_iter=1000, eps=0.01, bias=False, project=None):
        """ :loss: fonction de cout
            :loss_g: gradient de la fonction de cout
            :max_iter: nombre d'iterations
            :eps: pas de gradient
        """
        self.max_iter, self.eps = max_iter, eps
        self.loss, self.loss_g = loss, loss_g
        self.bias = bias
        self.project = project
        self.base = None

    # A general function to add projections
    def projection(self, datax):
        res = datax
        
        # Si on souhaite une projection gaussienne
        if (self.project == "gauss"):
            res = np.zeros([datax.shape[0], self.base.shape[0]])
            for i, x in enumerate(datax):
                for j, b in enumerate(self.base):
                    res[i,j] = np.exp(-np.linalg.norm(x-b)**2/0.1)
        
        # Transformation polynomiale (seulement pour 2 attributs)     
        if (self.project == "polynomial" and datax.shape[1]==2):
            res = np.column_stack([datax[:,0], datax[:,1], datax[:,0]*datax[:,1], datax[:,0]**2, datax[:,1]**2])

        # Si on souhaite ajouter un biais
        if(self.bias):
            vector = np.ones((res.shape[0],1))
            res = np.column_stack([vector, res])

        return res

    # Méthode d'apprentissage
    def fit(self,datax,datay,testx=None,testy=None):
        """ :datax: donnees de train
            :datay: label de train
            :testx: donnees de test
            :testy: label de test
        """
        datax = self.projection(datax)
        datay = datay.reshape(-1,1)
        datax = datax.reshape(len(datay),-1)
        self.w = np.random.random((1,datax.shape[1]))
        
        w_histo, f_histo, grad_histo = self.w, self.loss(datax, datay, self.w), self.loss_g(datax, datay, self.w)
        

        for i in range(self.max_iter):
            self.w = self.w - self.eps * self.loss_g(datax, datay, self.w)
            w_histo = np.vstack((w_histo, self.w))
            f_histo = np.vstack((f_histo, self.loss(datax, datay, self.w)))
            grad_histo = np.vstack((grad_histo, self.loss_g(datax, datay, self.w)))
        
        return w_histo, f_histo, grad_histo

    # Méthode pour obtenir la valeur prédite par le modèle
    def predict(self, datax):
        datax = self.projection(datax)
        if len(datax.shape)==1:
            datax = datax.reshape(1,-1)
        return np.sign(np.dot(datax, self.w.T)).reshape(-1)

    # Score la réussite avec le vrai label et des exemples à classer
    def score(self, datax, datay):
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
    plt.figure()
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()
    
def plot_trajectory(datax,datay,perceptron,step=10):
    plt.figure()
    w_histo, f_histo, grad_histo = perceptron.fit(datax,datay)
    xmax, xmin = np.max(w_histo[:,0]), np.min(w_histo[:,0])
    ymax, ymin = np.max(w_histo[:,1]), np.min(w_histo[:,1])
    dev_x, dev_y = abs(xmax-xmin)/4, abs(ymax-ymin)/4 # defines a margin for border
    dev_x += int(dev_x == 0)*5 # avoid dev_x = 0
    dev_y += int(dev_y == 0)*5
    grid,x1list,x2list=make_grid(xmin=xmin-dev_x,xmax=xmax+dev_y,ymin=ymin-dev_y,ymax=ymax+dev_y)
    plt.contourf(x1list,x2list,np.array([perceptron.loss(datax,datay,w)\
                                         for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.scatter(w_histo[:,0], w_histo[:,1], marker='+', color='black')
    plt.show()

def main():

    # Création des données selon deux distributions gaussiennes
    plt.ion()
    trainx,trainy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    testx,testy = gen_arti(nbex=1000, data_type=0, epsilon=1)
    # Base pour projection en distance gaussienne :
    base = trainx[np.random.randint(trainx.shape[0], size=100),:]
    
    # Tracé de l'isocontour de l'erreur pour MSE et HINGE
    plot_error(trainx,trainy,mse)
    plot_error(trainx,trainy,hinge)
    
    # Modèle standard avec MSE
    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.01)
    perceptron.fit(trainx,trainy)
    print("\nErreur mse : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()
    
    # Modèle standard Hinge
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(trainx,trainy)
    print("\nErreur hinge : train %f, test %f"% (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()
    
    # Modèle MSE avec biais
    perceptron = Lineaire(mse,mse_g,max_iter=1000,eps=0.01, bias=True)
    perceptron.fit(trainx,trainy)
    print("\nErreur mse biais : train %f, test %f"\
          % (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()

    # Modèle hinge avec biais
    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.1, bias=True)
    perceptron.bias = True
    perceptron.fit(trainx,trainy)
    print("\nErreur hinge biais : train %f, test %f"\
          % (perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()

    """ Beware when un-commenting : those models run very slowly """
    
    # Modèle mse avec projection gaussienne
    perceptron = Lineaire(mse, mse_g, max_iter=1000, eps=0.01, bias=False, project="gauss")
    perceptron.base = base
    perceptron.fit(trainx,trainy)
    print("\nErreur mse gauss : train %f, test %f"\
          %(perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()

    # Modèle hinge avec projection gaussienne
    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.5, bias=False, project="gauss")
    perceptron.base = base
    perceptron.fit(trainx,trainy)
    print("\nErreur hinge gauss : train %f, test %f"\
          %(perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()

    # Modèle mse avec projection polynomiale
    perceptron = Lineaire(mse, mse_g, max_iter=100, eps=0.0005, bias=False, project="polynomial")
    perceptron.fit(trainx,trainy)
    print("\nErreur mse polynomial : train %f, test %f"\
          %(perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()

    # Modèle hinge avec projection polynomiale
    perceptron = Lineaire(hinge, hinge_g, max_iter=100, eps=0.0005, bias=False, project="polynomial")
    perceptron.fit(trainx,trainy)
    print("\nErreur hinge polynomial : train %f, test %f"\
          %(perceptron.score(trainx,trainy),perceptron.score(testx,testy)))
    plt.figure()
    plot_frontiere(trainx,perceptron.predict,200)
    plot_data(trainx,trainy)
    plot_trajectory(trainx,trainy,perceptron)
    plt.show()
    
    
    """
    # Données USPS
    datax_train, datay_train = load_usps("USPS_test.txt")
    datax_test, datay_test = load_usps("USPS_train.txt")
    perceptron = Lineaire(hinge,hinge_g,max_iter=1000,eps=0.1)
    perceptron.fit(datax_train,datay_train)
    print("Erreur : train %f, test %f"% (perceptron.score(datax_train,datay_train),\
                                         perceptron.score(datax_test,datay_test)))

    #6 vs 9
    two_class_datax_train = datax_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    two_class_datay_train = datay_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    labely_train = np.sign(two_class_datay_train - 7)
    two_class_datax_test = datax_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    two_class_datay_test = datay_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    labely_test = np.sign(two_class_datay_test - 7)

    perceptron.fit(two_class_datax_train, labely_train)
    print("Erreur 2 classes 6/9: train %f, test %f"% (perceptron.score(two_class_datax_train,labely_train),\
                                                      perceptron.score(two_class_datax_test,labely_test)))

    #1 vs 8
    two_class_datax_train = datax_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    two_class_datay_train = datay_train[np.where(np.logical_or(datay_train == 1,datay_train == 8))]
    labely_train = np.sign(two_class_datay_train - 2)
    two_class_datax_test = datax_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    two_class_datay_test = datay_test[np.where(np.logical_or(datay_test == 1,datay_test == 8))]
    labely_test = np.sign(two_class_datay_test - 2)

    perceptron.fit(two_class_datax_train, labely_train)
    print("Erreur 2 classes 1/8: train %f, test %f"% (perceptron.score(two_class_datax_train,labely_train),\
                                                      perceptron.score(two_class_datax_test,labely_test)))

    #Uncomment to Plot the weights
    #print(*perceptron.w, sep='\n')
    #6 vs all
    labely_train = 2 * (datay_train == 6) - 1
    labely_test = 2 * (datay_test == 6) - 1
    perceptron.fit(datax_train, labely_train)
    print("Erreur one vs all: train %f, test %f"% (perceptron.score(datax_train,labely_train),\
                                                   perceptron.score(datax_test,labely_test)))
    """

if __name__=="__main__":
    main()