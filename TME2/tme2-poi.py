import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


plt.ion()
parismap = mpimg.imread('data/paris-48.806-2.23--48.916-2.48.jpg')

## coordonnees GPS de la carte
xmin,xmax = 2.23,2.48   ## coord_x min et max
ymin,ymax = 48.806,48.916 ## coord_y min et max

def show_map():
    plt.imshow(parismap,extent=[xmin,xmax,ymin,ymax],aspect=1.5)
    ## extent pour controler l'echelle du plan

poidata = pickle.load(open("data/poi-paris.pkl","rb"))
## liste des types de point of interest (poi)
print("Liste des types de POI" , ", ".join(poidata.keys()))

## Choix d'un poi
typepoi = "night_club"

## Creation de la matrice des coordonnees des POI
geo_mat = np.zeros((len(poidata[typepoi]),2))
for i,(k,v) in enumerate(poidata[typepoi].items()):
    geo_mat[i,:]=v[0]

## Affichage brut des poi
show_map()
## alpha permet de regler la transparence, s la taille
plt.scatter(geo_mat[:,1],geo_mat[:,0],alpha=0.8,s=3)


###################################################

# discretisation pour l'affichage des modeles d'estimation de densite
steps = 100
xx,yy = np.meshgrid(np.linspace(xmin,xmax,steps),np.linspace(ymin,ymax,steps))
grid = np.c_[xx.ravel(),yy.ravel()]

####################################################

# HISTOGRAMMES-

####################################################


res, _, _ = np.histogram2d(geo_mat[:,0], geo_mat[:,1], bins=15)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
plt.colorbar()






####################################################

# METHODE DES NOYAUX-

####################################################
count = 0
def parzen(x, y, h):
    global geo_mat
    global count
    count += 1
    if ((count % 10000) == 0):
        print(count)
    s = 0
    for i in range(len(geo_mat)):
        (x_i,y_i) = geo_mat[i]
        if (abs(x-x_i) < h/2 and abs(y-y_i) < h/2):
            s += 1       
    return (s / (len(geo_mat) * h))

parzen(48.3845, 2.35,2)

nb_pix_x, nb_pix_y, _ = np.shape(parismap)
xedges = np.linspace(xmin, xmax, nb_pix_x // 100)
yedges = np.linspace(ymin, ymax, nb_pix_y // 100)

def kernel_method(kernel, h):
    global nb_pix_x, nb_pix_y, xedges, yedges
    mat = np.zeros((nb_pix_x // 100, nb_pix_y // 100))
    for x in range(nb_pix_x // 100):
        for y in (range(nb_pix_y // 100)):
            mat[x,y] = kernel(yedges[y], xedges[x], h)
            print(yedges[y], xedges[x], h, mat[x,y])
    return mat

res = kernel_method(parzen,0.01)
plt.figure()
show_map()
plt.imshow(res,extent=[xmin,xmax,ymin,ymax],interpolation='none',\
               alpha=0.3,origin = "lower")
#imgplot = plt.imshow(mat)
plt.colorbar()

####################################################

# Que se passe-t-il pour une faible/forte discrétisation 
# pour la méthode des histogrammes ?

####################################################

####################################################

# • Quel est le rôle des paramètres des méthodes à noyaux ?

####################################################

####################################################

# Comment choisir de manière automatique les meilleurs paramètres ?

####################################################

####################################################

# La question reliée : comment estimer la qualité de votre modèle ?

####################################################