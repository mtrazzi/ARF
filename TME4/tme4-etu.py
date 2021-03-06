# AUTHORS: Michael Trazzi (mtrazzi) and Julien Denes (jdenes)

from arftools import plot_data, plot_frontiere, make_grid, gen_arti
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

########################################
# Fonctions utilitaires
########################################

# Decorateur pour transformer les vecteurs en matrices
def decorator_vec(fonc):
    def vecfonc(datax,datay,w,*args,**kwargs):
        if not hasattr(datay,"__len__"):
            datay = np.array([datay])
        datax,datay,w =  datax.reshape(len(datay),-1),datay.reshape(-1,1),w.reshape((1,-1))
        return fonc(datax,datay,w,*args,**kwargs)
    return vecfonc

# Importer les données USPS
def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

# Afficher une image de USPS
def show_usps(data):
    plt.figure()
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")
    plt.colorbar()
    plt.show()

# Affichage des isocourbes d'erreur des fonctions
def plot_error(datax,datay,f,step=10):
    plt.figure()
    grid,x1list,x2list=make_grid(xmin=-4,xmax=4,ymin=-4,ymax=4)
    plt.contourf(x1list,x2list,np.array([f(datax,datay,w) for w in grid]).reshape(x1list.shape),25)
    plt.colorbar()
    plt.show()

# Affichage des trajectoires d'optimisation sur les isocourbes des fonctions
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

# Affichage graphique d'un vecteur ou d'une matrice, typiquement les poids d'un perceptron
def plot_vector (w):
    img = plt.imshow(w, cmap = "viridis", interpolation = 'none')
    plt.colorbar()
    plt.show(img)

# Affichage des courbes d'apprentissages du perceptron
def plot_learning_curve(trainx, trainy, testx, testy, start=0, stop=1001, step=10):
    err_train, err_test = [], []
    iterations = list(range(start, stop, step))
    for i in iterations:
        perceptron = Lineaire(hinge, hinge_g, max_iter=i, eps=0.1)
        perceptron.fit(trainx, trainy)
        err_train.append(perceptron.score(trainx, trainy))
        err_test.append(perceptron.score(testx, testy))  
    plt.plot(iterations, err_train, color='blue', label="erreur (train)")
    plt.plot(iterations, err_test, color='red', label="erreur (test)")
    plt.legend()
    plt.show()

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

    # Une fonction pour ajouter une projection aux données
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


########################################
# Main
########################################

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
    

    # Attention : l'affichage de la frontières avec projections sont très longues à générer
    """
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

    # Modèle hinge avec projection polynomiale
    perceptron = Lineaire(hinge, hinge_g, max_iter=100, eps=0.1, bias=False, project="polynomial")
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
    
    #6 vs 9    
    trainx = datax_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    trainy = datay_train[np.where(np.logical_or(datay_train == 6,datay_train == 9))]
    labely_train = np.sign(trainy - 7)
    testx = datax_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    testy = datay_test[np.where(np.logical_or(datay_test == 6,datay_test == 9))]
    labely_test = np.sign(testy - 7)

    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.1)
    perceptron.fit(trainx, labely_train)
    print("Erreur 2 classes 6/9: train %f, test %f"% (perceptron.score(trainx,labely_train),\
                                                      perceptron.score(testx,labely_test)))

    plot_vector(perceptron.w.reshape(16,16))
    plot_learning_curve(trainx, labely_train, testx, labely_test, start=0, stop=501, step=10)
        
    #6 vs all
    labely_train = 2 * (datay_train == 6) - 1
    labely_test = 2 * (datay_test == 6) - 1
    
    perceptron = Lineaire(hinge, hinge_g, max_iter=1000, eps=0.1)
    perceptron.fit(datax_train, labely_train)
    print("Erreur one vs all: train %f, test %f"% (perceptron.score(datax_train,labely_train),\
                                                   perceptron.score(datax_test,labely_test)))
    
    # Attention : la courbe d'apprentissage est très longue à générer
    """
    plot_vector(perceptron.w.reshape(16,16))
    plot_learning_curve(datax_train, labely_train, datax_test, labely_test, start=0, stop=501, step=10)
    """
    
if __name__=="__main__":
    main()
    