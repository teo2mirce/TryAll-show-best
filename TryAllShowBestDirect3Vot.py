import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC



from sklearn.tree import ExtraTreeClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble.voting_classifier import VotingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import Perceptron
from sklearn.mixture import DPGMM
from sklearn.mixture import GaussianMixture
from sklearn.mixture import VBGMM

#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#todo, sa vad vecinii aia naspa


# f = open("es_train_data.csv")
f = open("Train.csv")
Features=f.readline().split(',')#It gives head
data = np.loadtxt(f,delimiter=",")
Features.pop(0)#primu e clasa
X_Train=data[:, 1:]
Y_Train=data[:,0]



# f = open("es_dev_data.csv")
f = open("Test.csv")
Features=f.readline().split(',')#It gives head
data = np.loadtxt(f,delimiter=",")
Features.pop(0)#primu e clasa
X_Test=data[:, 1:]
Y_Test=data[:,0]



models = []
models.append(('LR1', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('Knn1',     KNeighborsClassifier(1) ))
models.append(('Knn9D',     KNeighborsClassifier(9, weights='distance') ))
models.append(('LSVM',     SVC(kernel="linear")  ))
models.append(('RBF',     SVC()  ))
models.append(('DT',     DecisionTreeClassifier(max_depth=5) ))
models.append(('RF',     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1) ))
models.append(('NN1',     MLPClassifier(alpha=0.1) ))
models.append(('AB',     AdaBoostClassifier() ))
models.append(('NB',     GaussianNB() ))
models.append(('QDA',     QuadraticDiscriminantAnalysis()  ))
models.append(('NuSVC',     NuSVC(probability=True)  ))
models.append(('GBC',     GradientBoostingClassifier()  ))
models.append(('Q2',     BaggingClassifier()  ))


models.append(('RBF2',     SVC(kernel="rbf", C=0.025, probability=True)  ))
models.append(('ETC',     ExtraTreeClassifier()  ))
models.append(('Q1',     SGDClassifier()  ))
models.append(('Q2',     RidgeClassifier()  ))
models.append(('Q3',     PassiveAggressiveClassifier()  ))
models.append(('Q4',     GaussianProcessClassifier()  ))
models.append(('Q5',     ExtraTreesClassifier()  ))
models.append(('Q6',     BernoulliNB()  ))
models.append(('Q7',     GaussianMixture()  ))


#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=250)
forest.fit(X_Train, Y_Train)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

print('Important features: ',np.array(Features)[indices])
# indices=[ 0,2,3,4,5,7,8,9,14,23,32,36,38,39,40]
from random import shuffle
import random
# shuffle(indices)


from collections import Counter

Predictii=[  []  for _ in range(len(Y_Test))]
	
Accs=[]
for normalizare in ["Fara","N1","N2"]:

	if(normalizare=="Fara"):
		X_Test_N=X_Test
		X_Train_N=X_Train
	if(normalizare=="N1"):
		X_Train_N=normalize(X_Train)
		X_Test_N=normalize(X_Test)
	if(normalizare=="N2"):
		m=len(Features)
		mins=[ min(X_Train[:,col].min(),X_Test[:,col].min()) for col in range(m)]
		maxs=[ max(X_Train[:,col].max(),X_Test[:,col].max()) for col in range(m)]
		for j in range(m):
			X_Test_N=X_Test
			X_Train_N=X_Train
			X_Train_N[:,j]=(X_Train[:,j]-mins[j])/(maxs[j]-mins[j]+1)
			X_Test_N[:,j]=(X_Test[:,j]-mins[j])/(maxs[j]-mins[j]+1)
	
	for name, model in models:
		start_time = time.time()
		model.fit(X_Train_N,Y_Train)
		Preds=model.predict(X_Test_N)
		acc=(Preds==Y_Test).mean()
		print(normalizare,' ',name,' ',acc,' time: ',time.time() - start_time)
						
		if len(Accs)<=10 or acc>=np.array(Accs).mean():
			print('adaugat')
			Accs.append(acc)
			for i in range(0,len(Preds)):
				Predictii[i].append(Preds[i])
		
		
		

BestIndex=np.array([x for x in Accs]).argsort()[::-1][:10]

for i in range(len(Predictii)):
	Pred=np.array(Predictii[i])[BestIndex]
	print(X_Test[i],'->',Counter(Pred).most_common(1)[0][0],' ',100.0*Counter(Pred).most_common(1)[0][1]/(len(Pred)),'%')
