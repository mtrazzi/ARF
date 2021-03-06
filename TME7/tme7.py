import numpy as np
#needed: derivee par rapport au parametres et derivee de la sortie par rapport aux entrees
#modules:
#1) module (eg lineaire) : ok
#2) module activation : pas  de parametres, seulement derivee

class Loss(object):
	def forward(self, y, yhat):
		return
		#cout
	def backward(self, y, yhat):
		return
		#calcul le gradient du cout

#############################NOTES

## DEUX TYPES DE MODULES: Activation et Lineires

## Pour les fonctions d'activation: pas de parametres w_ij, seulement une derivee

## En gros, c'est ton g' (2 si tu transformes ton erreur en faisant *2 partout)


class SigmoidModule(object):
	#
	def __init__(self):
		self._parameters = None #matrice de poids par exemple
		self._gradient = None #accumule le gradient calcule
	def zero_grad(self):
		self._gradient = np.zeros(self._gradient.shape())
		#Annule gradient
	def forward(self, X):
		np.dot(w.T, X)
		return
		#passe forward
	def update_parameters(self, gradient_step):
		return
		#mise a jour des parametres selon le gradient et le pas gradient_step
	def backward_update_gradient(self, input, delta):
		return
		#met a jour la valeur du gradient par rapport au parametres
	def backward_delta(self, input, delta):
		return
		#derivee de l'erreur : renvoie l'erreur derriere

class LinearModule(object):
	def __init__(self,n,m):
		self._parameters = np.zeros((n,m)) #matrice de poids par exemple
		self._gradient = np.zeros((n,m))#accumule le gradient calcule
	def zero_grad(self):
		#Annule gradient
		self._gradient = np.zeros(self._gradient.shape())
	def forward(self, X):
		return np.dot(self._parameters.T, X.T).T
		#passe forward
	def update_parameters(self, gradient_step):
		self._parameters -= gradient_step * self._gradient
		return
		#mise a jour des parametres selon le gradient et le pas gradient_step
	def backward_update_gradient(self, input, delta):
		# EN GROS: accumulation du gradient, qui est la derivee de la Loss par rapport aux poids
		# Donc une matrice de taille de w_ij pour i,j qui varient bien pour les poids
		# On accumule le gradient pour pouvoir faire du stochastique ou du batch
		# C'est pas optimise, il faudrait plutot faire du calcul avec des tenseurs plutot que de faire des appels systemes a chaque fois
		#self._gradient = self.gradi np.dot(input, delta)
		return
		#met a jour la valeur du gradient par rapport au parametres
	def backward_delta(self, input, delta):
		delta * 
		return
		#derivee de l'erreur : renvoie l'erreur derriere

def main():
	datax = np.random.random_sample((1,5))
	m = LinearModule(5,1)
	result = m.forward(datax)
	m.backward_update_gradient(datax, [0,0,0,0])
	print(result)
	return

if __name__ == "__main__":
	main()
