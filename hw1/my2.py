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

IMP note: to elementwise multiply, you better use the type ‘ndarray’ (maybe not 'ndmatrix').
and the shape should be consistent: (3,)*(3,)=(3,); (3,1)*(3,1)=(3,1);but (3,)*(3,1)is (3,3)
it's always good practice to reshape a defult 1-d array, for example, use reshape() to convert (3,) to (3,1)
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

#gradient for SGD, so take x as an array
def compute_gradient(x,yt,yp):
    cost = yt - yp #got a (1,1)ndarray
    gradient = x.T.dot(cost)
    return gradient.T

#take the whole x
def log_likelihood(x, yt, b):
    yt = np.asarray(yt).reshape(x.shape[0],1) #ndarray,shape(x,1)
    # print ('yt shape',yt.shape,type(yt))
    z = x.dot(b.T) # ndarray, shape(x,1)
    # print ('z shape',z.shape, type(z))
    ll = np.sum( yt * z - np.log(1 + np.exp(z)) )
    # aa = yt *z
    # bb = np.log(1 + np.exp(z))
    # print ('shape of y*z',aa.shape,type(aa))
    # print ('shape of np.log()',bb.shape,type(bb))
    return ll

def logistic_regression(x, yt, learning_rate, num_steps):
    b = np.zeros((1,x.shape[1]))
    # print ('initate b, the shape', b.shape)
    lst_index = [row for row in range(x.shape[0])]
    shuffle(lst_index)
    for step in range(num_steps):
        i = step % x.shape[0]
        i = lst_index[i]
        xi = x.getrow(i)
        zi = xi.dot(b.T)
        ypi = sigmoid(zi)
        yti = yt[i]
        gradient = compute_gradient(xi,yti,ypi)
        b += learning_rate * gradient

        # if step == 0: #debug
        #     print ('shape of x1', xi.shape)
        #     print ('value of zi: ', zi, 'shape',zi.shape)
        #     print ('value of ypi: ', ypi, 'shape',ypi.shape)
        #     print ('value of gradient: ', gradient, 'shape',gradient.shape)
        #     print ('value of update b: ', b, 'shape',b.shape)

        lst_step = []
        lst_ll = []
        if (step+1) % 10000 == 0:
            lst_step.append(step)
            ll = log_likelihood(x, yt, b)
            lst_ll.append(ll)
            print ('step=',step, ',  ll=', ll)
    return {'b':b,'lst_step':lst_step,'lst_ll':lst_ll}

def predict(x, b):
    z = x.dot(b.T)
    yp = sigmoid(z)
    ytr = np.round(yp)
    ytr.astype(int)
    return ytr

def f1(yt,yp):
    return f1_score(yt,yp,average='micro')

lst = get_x_y_matrix('train.tsv',with_label = True)
train_x = lst[0]
train_yt = lst[1]
train_vo = lst[2]
# print ('shape of train_x: ', train_x.shape)
# print ('len of train yt', len(train_yt))
# print ('len of tain_vo: ', len(train_vo))
dct = logistic_regression(train_x, train_yt,learning_rate = 5e-5, num_steps = 300000)
b = dct['b']
# print ('final b shape: ', b)
lst_step = dct['lst_step']
lst_ll = dct['lst_ll']
plt.plot(lst_a,lst_f1)


#on dev.tsv
dct = predict_multi('dev.tsv',True, b, train_vo)
lst_yp = dct['lst_yp']
lst_yt = dct['lst_yt']
# print (lst_yp)

F1 = f1(lst_yt,lst_yp)
print (F1)

#on unlabeled test.tsv
# dct = predict_multi('test.unlabeled.tsv',False, b, train_vo)
# lst_yp = dct['lst_yp']
# lst_yt = dct['lst_yt']
