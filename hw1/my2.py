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
    # with open('stopwords.txt','r') as f:
    #     lines = f.readlines()
    #     stop_words = []
    #     for line in lines:
    #         line = line.strip('\n')
    #         stop_words.append(line)
        # print (stop_words)

    line = inst.lower()
    lst_1 =line.split()
    lst_2 = list()

    stopwords = ['', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been','before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

    p1 = r'http.+'
    p2 = r'@.*'
    p3 = r'&#.*'
    p4 = r'[a-z]+.*[a-z]+(?=[^a-z]+)' #end with non-cha
    for w in lst_1:
        w = re.sub(p1,'',w)
        w = re.sub(p2,'',w)
        w = re.sub(p3,'',w)
        m = re.search(p4,w)
        if m:
            w = m.group(0)
        if w not in stopwords:
            lst_2.append(w)
    return lst_2

# def tokenize(inst):
#     line = inst.lower()
#     lst_1 =line.split()
#     lst_2 = list()
#     pattern =r'[a-z0-9]+'
#     pattern2 = r'[a-z0-9]+\'[a-z0-9]+'
#     pattern3 = r'[!.?]+'
#     for w in lst_1:
#         match2 = re.search(pattern2,w)
#         match = re.search(pattern,w)
#         match3 = re.search(pattern3,w)
#         if match2:
#             lst_2.append(match2.group(0))
#         elif match:
#             lst_2.append(match.group(0))
#         elif match3:
#             lst_2.append(match3.group(0))
#     return lst_2
# def tokenize(inst):
#     line = inst.lower()
#     lst_1 =line.split()
#     lst_2 = list()
#
#     for w in lst_1:
#         p1 =r'[a-z]+'
#         p2 = r'[0-9]+'
#         p3 = r'[a-z]+\'[a-z]+'
#         p4 = r'[!?]+'
#         p5 = r'\.\.+'
#         m1 = re.findall(p1,w)
#         m2 = re.findall(p2,w)
#         m3 = re.findall(p3,w)
#         m4 = re.findall(p4,w)
#         m5 = re.findall(p5,w)
#         if len(m3)>0:
#             for i in m3:
#                 lst_2.append(i)
#         else:
#             for i in m1:
#                 lst_2.append(i)
#         for i in m2:
#             lst_2.append(i)
#         for i in m4:
#             lst_2.append(i)
#         for i in m5:
#             lst_2.append(i)
#     return lst_2
# def tokenize(inst):
#     lst = inst.split()
#     return lst

#get x and y from training data
def get_x_y_matrix(file, with_label,vocabulary = {}):
    with open(file,'r', encoding= 'utf-8') as f:
        rdata = []
        reader = f.readlines()[1:]
        for line in reader:
            rdata.append(re.split("\t",line.replace('\n','')))

        yt = []
        indices = []
        data = []
        indptr = [0]

        for row in rdata:
            if with_label:
                yt.append(int(row[2]))

            lst_term = tokenize(row[1])
            for term in lst_term:
                index = vocabulary.setdefault(term, len(vocabulary))
                indices.append(index)
                data.append(1)
            indptr.append(len(indices))

    x = csr_matrix((data, indices, indptr))
    intercept = np.ones((x.shape[0], 1))
    x = hstack((intercept, x))

    return [x,yt,vocabulary]

def sigmoid(z):
    # print()
    # print('sigmoid')
    # print ('z',type(z),z.shape)
    yp = 1/(1+np.exp(-z)) #(1,1)
    return yp

#gradient for SGD, so take x as an array
def compute_gradient(x,yt,yp):
    # print('compute_gradient')
    cost = yt - yp #got a (1,1)ndarray
    # print ('cost',type(cost),cost.shape)
    gradient = x.T.dot(cost)
    # print ('gradient',type(gradient),gradient.shape)
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
    lst_step = []
    lst_ll = []
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
        if (step+1) % 10000 == 0:
            lst_step.append(step+1)
            ll = log_likelihood(x, yt, b)
            lst_ll.append(ll)
            print ('step=',step+1, ',  ll=', ll)
    return {'b':b,'lst_step':lst_step,'lst_ll':lst_ll}

def predict(x, b):
    z = x.dot(b.T)
    yp = sigmoid(z)
    # print ('type yp',type(yp),'shape',yp.shape)
    ytr = np.round(yp) #[[x]]
    return ytr

def predict_multi(file, with_label,b, vocabulary):
    with open(file,'r', encoding= 'utf-8') as f:
        rdata = []
        reader = f.readlines()[1:]
        for line in reader:
            rdata.append(re.split("\t",line.replace('\n','')))

        yt = []
        lst_yp = []
        lst_id = []
        for row in rdata: #compute y-predict for each row
            lst_id.append(row[0])
            if with_label:
                yt.append(int(row[2]))

            x = np.zeros((1,len(vocabulary)+1)) #initiate a vector, including the column of bias
            x[0,0]=1 #bias set to 1
            lst_term = tokenize(row[1])
            for term in lst_term:
                if term in vocabulary:
                    index = vocabulary[term]
                    x[0,index] += 1

            yp = int(predict(x,b)) #turn [[x]] to x
            lst_yp.append(yp)


    return {'lst_yp':lst_yp,'lst_yt':yt,'lst_id':lst_id}


def f1(yt,yp):
    return f1_score(yt,yp,average='micro')

lst = get_x_y_matrix('train.tsv',with_label = True)
train_x = lst[0]
train_yt = lst[1]
train_vo = lst[2]

# print ('shape of train_x: ', train_x.shape)
# print ('len of train yt', len(train_yt))
# print ('len of tain_vo: ', len(train_vo))
# dct = logistic_regression(train_x, train_yt,learning_rate = 5e-5, num_steps = 300000)
# lst_ll = dct['lst_ll']
# dct = logistic_regression(train_x, train_yt,learning_rate = 5e-6, num_steps = 300000)
# lst_ll_3 = dct['lst_ll']
dct = logistic_regression(train_x, train_yt,learning_rate = 5e-3, num_steps = 400000)
lst_ll_2 = dct['lst_ll']
lst_step = dct['lst_step']
b = dct['b']



# print ('final b shape: ', b)
plt.plot(lst_step,lst_ll,label='lr=5e-5')
plt.plot(lst_step,lst_ll_2,label='lr=5e-3')
plt.plot(lst_step,lst_ll_3,label = 'lr=5e-6')
plt.legend()
plt.savefig('steps_lr.png')
plt.close()

#on dev.tsv
dct_p = predict_multi('dev.tsv',True, b, train_vo)
lst_yp = dct_p['lst_yp']
lst_yt = dct_p['lst_yt']
# print (lst_yp)

F1 = f1(lst_yt,lst_yp)
print (F1)

#on unlabeled test.tsv
dct = predict_multi('test.unlabeled.tsv',False, b, train_vo)
lst_yp = dct['lst_yp']
lst_id = dct['lst_id']

with open('test2.csv', 'w') as test2:
    test2.write("instance_id,class\n")
    for i in range(len(lst_yp)):
        test2.write(lst_id[i]+","+str(lst_yp[i])+"\n")
