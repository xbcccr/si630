from scipy.sparse import csr_matrix
import csv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.sparse import hstack
from random import shuffle


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
    pattern3 = r'[!.?]+'
    for w in lst_1:
        match2 = re.search(pattern2,w)
        match = re.search(pattern,w)
        match3 = re.search(pattern3,w)
        if match2:
            lst_2.append(match2.group(0))
        elif match:
            lst_2.append(match.group(0))
        elif match3:
            lst_2.append(match3.group(0))
    return lst_2

#get x and y from training data
def get_x_y_matrix(file, with_label,vocabulary = {}):
    with open(file) as data:
        reader2 = csv.DictReader(data, dialect='excel-tab')

        yt = []
        indices = []
        data = []
        indptr = [0]

        for row in reader2:
            if with_label:
                yt.append(int(row['class']))

            lst_term = tokenize(row['text'])
            for term in lst_term:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

    x = csr_matrix((data, indices, indptr))
    intercept = np.ones((x.shape[0], 1))
    x = hstack((intercept, x))

    return [x,yt,vocabulary]

def predict_multi(file, with_label,b, vocabulary):
    with open(file) as data:
        reader2 = csv.DictReader(data, dialect='excel-tab')

        yt = []
        lst_yp = []
        for row in reader2: #compute y-predict for each row
            if with_label:
                yt.append(int(row['class']))

            x = np.zeros((1,len(vocabulary)+1)) #initiate a vector, including the column of bias
            x[0,0]=1 #bias set to 1
            lst_term = tokenize(row['text'])
            for term in lst_term:
                if term in vocabulary:
                    index = vocabulary[term]
                    x[0,index] += 1

            yp = int(predict(x,b))
            lst_yp.append(yp)


    return {'lst_yp':lst_yp,'lst_yt':yt}


def sigmoid(z):
    return 1/(1+np.exp(-z))

#take the whole x
def log_likelihood(x, yt, b):
    yt = np.asarray(yt).reshape(x.shape[0],1) #ndarray,shape(x,1)
    z = x.dot(b.T) # a (x,1)matrix
    print ('z shape',z.shape, 'z type',type(z))
    ll = np.sum( yt * z - np.log(1 + np.exp(z)) )
    print ('yt[0]',yt[0],'shape',yt.shape)

    print ('shape of np.log()',np.log(1 + np.exp(z)).shape)
    a = yt*z
    print (type(a))
    print ('shape of yt*z',a.shape)
    return ll

#gradient for SGD, so take x as an array
def compute_gradient(x,yt,yp):
    cost = yt - yp #got a (1,1)matrix
    gradient = x.T.dot(cost)
    return gradient.T

def logistic_regression(x, yt, learning_rate, num_steps):
    b = np.zeros((1,x.shape[1]))
    for step in range(num_steps):
        print ('step',step)
        i = step
        i = lst_index[i]
        xi = x.getrow(i)
        zi = xi.dot(b.T) #a matrix
        ypi = sigmoid(zi) #a matrix
        yti = yt[i] #a number
        gradient = compute_gradient(xi,yti,ypi)
        b += learning_rate * gradient

        print ('x1', xi)
        print ('zi: ', zi)
        print ('ypi: ', ypi)
        print ('gradient: ', gradient)
        print ('updated b: ', b)

        ll = log_likelihood(x, yt, b)
        print ('ll=', ll)
        print()
    return b

def predict(x, b):
    z = x.dot(b.T)
    yp = sigmoid(z)
    ytr = np.round(yp)
    ytr.astype(int)
    return ytr

def f1(yt,yp):
    return f1_score(yt,yp,average='micro')

lst = get_x_y_matrix('s.tsv',with_label = True)
train_x = lst[0]
train_yt = lst[1]
train_vo = lst[2]
print (train_x, train_yt,train_vo)
# b = logistic_regression(train_x, train_yt,learning_rate = 5e-5, num_steps = 100000)
# print ('final b shape: ', b)

# on dev.tsv
# dct = predict_multi('s.tsv',True, b, train_vo)
# lst_yp = dct['lst_yp']
# lst_yt = dct['lst_yt']
# print (lst_yp)
#
# F1 = f1(lst_yt,lst_yp)
# print (F1)

#on unlabeled test.tsv
# dct = predict_multi('test.unlabeled.tsv',False, b, train_vo)
# lst_yp = dct['lst_yp']
# lst_yt = dct['lst_yt']
