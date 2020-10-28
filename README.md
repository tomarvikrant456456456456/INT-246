# INT-246
#uploading buddymove.csv on Google Colab
from google.colab import files
uploaded= files.upload()
import io
dataset= pd.read_csv('buddymove.csv')
dataset.head
dataset.shape
#assigning target and feature variable
features=['Picnic','Religious','Nature','Theatre','Shopping']
X=dataset[features]
Y=dataset.Sports
#splitting thhe data into 70% training and 30% testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)
#implementing perceptron
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

clf= Perceptron()
clf.fit(X_train, Y_train) #training data using perceptron
clf.score(X,Y)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
model=keras.Sequential(name="Back-Propagation")
model.add(keras.Input(shape=5,name="input_layer"))
model.add(layers.Dense(128,activation='sigmoid',name="Hidden1"))
model.add(layers.Dense(32,activation='sigmoid',name="Hidden2"))
model.add(layers.Dense(1,activation='sigmoid',name="output"))
model.compile( loss=keras.losses.binary_crossentropy,optimizer='rmsprop', metrics=['accuracy'])
output=model.fit(X_train,Y_train, epochs=50)
#implementing SVM
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
clf = svm.SVC(kernel='linear')
clf = clf.fit(X_train, Y_train)
o1=clf.predict(X_test)
accuracy_score(o1,Y_test) #checking accuracy
o2=clf.predict(X_train)
accuracy_score(o2,Y_train) #checking accuracy
support_vector_indices = clf.support_
print(support_vector_indices)  #print the index of all support vectors which will maximise the margin
#implementing SOM using Minisom
import numpy as np
dataset=np.array(dataset)
Z=dataset[:,2:]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
Z = sc.fit_transform(Z)
!pip install minisom
from minisom import MiniSom
som = MiniSom( x = 10, y = 10, input_len = 5, sigma = 1.0, learning_rate = 0.6)
#init the weight
som.random_weights_init(Z)
som.train_random(data = Z, num_iteration = 100)
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
for i, x in enumerate(Z):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()
#Implementing LVQ
import numpy as np
dataset=np.array(dataset)
X=dataset[:,2:] #data
Y=dataset[:,1] #target
lr=0.6 #learning rate parameter
w1=np.random.ranf(5,) #initialising random weights 
w2=np.random.ranf(5,)
#calculate euclidean distance
def distance(x,w):
    return(np.sqrt(np.sum((x-w)**2)))

def update_w1(x,w,lr):    # if Target=0
    return (w+lr*(x-w))

def update_w2(x,w,lr):    #if Target=1
    return (w-lr*(x-w))
    
for i in range(1):
  print("Iteration: ",i+1)
  for j in range(249):
    print("Input:", X[j])
    print("Target", Y[j])
    d1=distance(X[j],w1)
    d2=distance(X[j],w2)
    if d1<=d2:
      if Y[j]==0:
        w1=update_w1(X[j],w1,lr)
      else:
        w1=update_w2(X[j],w1,lr)
    else:
      if Y[j]==1:
        w2=update_w1(X[j],w2,lr)
      else:
        w2=update_w2(X[j],w2,lr)   
  lr=lr*0.8
print("Updated lr:",lr)


#Checking the classification
Sample = [ 35, 99, 201, 190, 195] 
dis1=distance(Sample,w1)
dis2=distance(Sample,w2)
if(dis1>dis2):
  print("Sample belongs to Class 1")
else:
  print("Sample belongs to Class 2")  
