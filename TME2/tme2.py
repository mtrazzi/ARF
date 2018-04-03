import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from scipy.stats import multivariate_normal

plt.interactive(False)
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    #plt.show()
    # extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
## print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

### Affichage brut des poi
#show_map()
### alpha permet de regler la transparence, s la taille
#plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

####################################################

# METHODE DES HISTOGRAMMES

####################################################

def histo(nb_bins):
    global geo_mat
    res, _, _ = np.histogram2d(geo_mat[:,0], geo_mat[:,1], bins=nb_bins)
    plt.figure()
    plt.title('Histogramme: ' + str(nb_bins) + ' subdivisions')
    show_map()
    plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
                   alpha=0.3,origin = "lower")
    plt.colorbar()
    plt.show()


####################################################

# METHODE DES NOYAUX

####################################################

# Fenetre de Parzen
def parzen(x, y, h, sigma=1): #sigma only for kernel_method
   global geo_mat
   s = 0
   for i in range(len(geo_mat)):
       (x_i,y_i) = geo_mat[i]
       if (abs(x-x_i) < h/2 and abs(y-y_i) < h/2):
           s += 1
   V = h * h
   return (s / (len(geo_mat) * V))

def gaussian(x, y, h, sigma=1):
    global geo_mat
    var = multivariate_normal(mean=[0, 0], cov=[[sigma, 0], [0, sigma]]) #to estimate the gaussian distribution
    s = 0
    for i in range(len(geo_mat)):
        (x_i, y_i) = geo_mat[i]
        s += var.pdf([(x-x_i)*100, (y-y_i)*100]) #multiply *100 to scale for the gaussian distribution
    return (s / (len(geo_mat) * h))

parzen(48.3845, 2.35,2)

nb_pix_x, nb_pix_y, _ = np.shape(parismap)
xedges = np.linspace(xmin, xmax, nb_pix_x // 100)
yedges = np.linspace(ymin, ymax, nb_pix_y // 100)

def kernel_method(kernel, h, sigma=1):
   global nb_pix_x, nb_pix_y, xedges, yedges
   mat = np.zeros((nb_pix_x // 100, nb_pix_y // 100))
   for x in range(nb_pix_x // 100):
       for y in (range(nb_pix_y // 100)):
           mat[x,y] = kernel(yedges[y], xedges[x], h, sigma)
##         print(yedges[y], xedges[x], h, mat[x,y])
   return mat

def main():

    ####################################################
    
    # discretisation pour la methode des histogrammes ?
    
    # Faible vs Forte discretisation
    
    ####################################################
    #for i in range(1,5):
    #    histo(5 * i)

    ####################################################
    
    # Quel est le role des parametres des methodes a noyaux ?
    
    ####################################################

    # PARZEN
    
    for h in [0.005, 0.01, 0.05]:    
        res = kernel_method(parzen,h)
        plt.figure()
        show_map()
        plt.title('Parzen: fenêtre de ' + str(h))
        plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
                      alpha=0.3,origin = "lower")
        plt.colorbar()
        plt.show()

    # GAUSSIAN
    h = 0.1
    for sigma in [0.1, 0.5, 1]:    
        res = kernel_method(gaussian,h, sigma)
        plt.figure()
        show_map()
        plt.title('Gaussian: sigma=' + str(sigma) + ' ,h=' + str(h))
        plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
                      alpha=0.3,origin = "lower")
        plt.colorbar()
        plt.show()

    ####################################################
    
    # Comment choisir de maniere automatique les meilleurs parametres ?
    
    ####################################################

    # Grid search
      
    ####################################################
    
    # La question reliee : comment estimer la qualite de votre modèle
    
    ####################################################

    # Voir si l'echantillonage selon la densité estimé correspond aux données

if  __name__ == "__main__":
    main()
