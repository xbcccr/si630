from scipy.sparse import csr_matrix
import csv
import numpy as np
from scipy.sparse import hstack

docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world",'wow']]
indptr = [0]
indices = []
data = []
vocabulary = {}
for d in docs:
     for term in d:
         index = vocabulary.setdefault(term, len(vocabulary))
         indices.append(index)
         data.append(1)
     indptr.append(len(indices))

x = csr_matrix((data, indices, indptr), dtype=int)


b_a = np.zeros((1,x.shape[1]), dtype=np.int)




test = [["goodbye", "cruel", "world",'wow'],["hello", "world", "hello"]]
indptr = [0]
indices = []
data = []
for d in test:
     for term in d:
         index = vocabulary[term]
         indices.append(index)
         data.append(1)
     indptr.append(len(indices))
y = csr_matrix((data, indices, indptr), dtype=int)

# print ((x[0]*2).toarray())
intercept = np.ones((x.shape[0], 1))
x_1 = hstack((intercept, x))
for i in range(2):
    xi = x_1.getrow(i)
    print (xi)
    print()

b = np.zeros((1,x.shape[1]),dtype = np.int)
z0 = x[0].dot(b.T)
print (z0)
yp0 = 1/(1+np.exp(-z0))
print (yp0)
print (yp0.shape)
cost = 1-yp0
print (cost.shape)
print ((x[0].T.dot(cost)))

e = [[1],[3],[4],[5]]
f = np.array(e)
g = f.T.tolist()
print (g)

hh = [1,2,4]
h = np.asarray(hh)
print (h.shape)

yt = [1,0,1]
yt = np.asarray(yt).T
print (yt)
yt
