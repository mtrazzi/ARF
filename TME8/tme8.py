import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from copy import copy, deepcopy
from random import randint, uniform


# Returns distance mean from center
def loss(X, prototype):
    return np.mean(cdist(X, prototype))

# Returns barycenter of a set of data
def find_prototype(cluster):
    if not len(cluster) == 0: return np.mean(cluster, axis = 0)
    else: return np.array([uniform(ranges[i][0], ranges[i][1]) for i in range(d)])

# Returns closest barycenter for eack point in X
def argmin_prototype(X, prototypes):
    argmins = np.argmin(cdist(X, prototypes), axis = 1)
    return argmins

def expectation(X, prototypes):
    expected_attributions = argmin_prototype(X, prototypes)
    clusters = [[] for p in prototypes]
    for i, x in enumerate(X):
        clusters[expected_attributions[i]].append(x)
    return np.array(clusters)

def maximisation(clusters):
    new_proto = np.zeros((len(clusters), d))
    for i in range(len(clusters)):
        new_proto[i] = find_prototype(clusters[i])
    return new_proto

def k_means(X, init_prototypes, max_iter=1000):
    prototypes = deepcopy(init_prototypes)
    old_proto = None
    i = 0
    while (not np.array_equal(old_proto, prototypes) and i < max_iter):
        # print(prototypes)
        i += 1
        old_proto = deepcopy(prototypes)
        clusters = expectation(X, prototypes)
        prototypes = maximisation(clusters)
    return prototypes, clusters

def compress(pixels, k):
    global d
    global ranges
    d = len(pixels[0])
    ranges = [(np.min(pixels[:,i]), np.max(pixels[:,i])) for i in range(d)]
    init_clusters = [[] for c in range(k)]
    # Initialize randomly
    for p in pixels:
        init_clusters[randint(0,k-1)].append(copy(p))
    init_prototypes = maximisation(np.array(init_clusters))
    # Launch clustering
    prototypes, clusters = k_means(pixels, init_prototypes, 30)
    # Transform pixels into their prototypes
    compressing = deepcopy(pixels)
    for i, p in enumerate(pixels):
        classif = False
        for k, c in enumerate([c for c in clusters if len(c) != 0]):
            if np.any(c == p):
                compressing[i] = prototypes[k]
                classif = True
                break
        if not classif: print("Mega error")
    return compressing
    

im = plt.imread("fichier2.png")[:,:,:3]
im_h, im_l, _ = im.shape
pixels = im.reshape((im_h * im_l, 3)) # in RGB
pixels_hsv = rgb_to_hsv(pixels) # in HSV
compressed_img = compress(pixels_hsv, 300)
final_rgb = hsv_to_rgb(compressed_img)
imnew = final_rgb.reshape((im_h, im_l, 3))
plt.imshow(imnew)
    
    
    