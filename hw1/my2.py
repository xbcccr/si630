from scipy.sparse import csr_matrix
import csv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.sparse import hstack


'''variable explanation
x:features matrix
b:weights matrix
z:bx, or scores
yt:lables/targets
yp:predictions(probability)
yc:classification(yc=0 or 1)
'''

def tokenize(inst):
    line = inst.lower()
    lst_1 =line.split()
    lst_2 = list()
    pattern =r'[a-z0-9]+'
    pattern2 = r'[a-z0-9]+\'[a-z0-9]+'
    for w in lst_1:
        match2 = re.search(pattern2,w)
        match = re.search(pattern,w)
        if match2:
            lst_2.append(match2.group(0))
        elif match:
            lst_2.append(match.group(0))
    return lst_2

#get x and y from training data
def get_x_y_matrix(file, vocabulary = {}):
    with open(file) as data:
        reader2 = csv.DictReader(data, dialect='excel-tab')

        yt = []

        indices = []
        data = []
        indptr = [0]

        for row in reader2:
            yt.append(int(row['class']))

            lst_term = tokenize(row['text'])
            for term in lst_term:
                index = vocabulary.setdefault(term, len(vocabulary))
                # index = vocabulary[term]
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

    x = csr_matrix((data, indices, indptr))
    intercept = np.ones((x.shape[0], 1))
    x = hstack((intercept, x))

    return [x,yt,vocabulary]

def sigmoid(z):
    return 1/(1+np.exp(-z))

#take the whole x
def log_likelihood(x, yt, b):
    yt = np.array(yt).T #a vector
    z = x.dot(b.T) # a vector
    ll = np.sum( yt * z - np.log(1 + np.exp(z)) )
    return ll

#gradient for SGD, so take x as an array
def compute_gradient(x,yt,yp):
    cost = yt - yp #got a (1,1)matrix
    gradient = x.T.dot(cost)
    return gradient.T

def logistic_regression(x, yt, learning_rate, num_steps):
    b = np.zeros((1,x.shape[1]))
    for step in range(num_steps):
        i = step % x.shape[0]
        xi = x.getrow(i)
        zi = xi.dot(b.T) #a matrix
        ypi = sigmoid(zi) #a matrix
        yti = yt[i] #a number
        gradient = compute_gradient(xi,yti,ypi)
        b += learning_rate * gradient

        lst_step = []
        lst_ll = []
        if step % 10000 == 0:
            lst_step.append(step)
            ll = log_likelihood(x, yt, b)
            lst_ll.append(ll)
            print ('step=',step, ',  ll=', ll)
    return b

def predict(x, b):
    z = x.dot(b.T)
    yp = sigmoid(z)
    yc = np.round(yp)
    yc.astype(int)
    return yc

def f1(yt,yc):
    return f1_score(yt,yc,average='micro')

lst = get_x_y_matrix('train.tsv')
train_x = lst[0]
train_yt = lst[1]
train_vo = lst[2]
print (train_x.shape)
b = logistic_regression(train_x, train_yt,learning_rate = 5e-5, num_steps = 300000)
lst_x_yt = get_x_y_matrix('dev.tsv',train_vo)
predict_x = lst_x_yt[0]
predict_yt = lst_x_yt[1]
yc = predict(predict_x,predict_yt)
lst_yc = list(yc)
F1 = f1(predict_yt,lst_yc)
