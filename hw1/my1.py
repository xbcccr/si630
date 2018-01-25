import csv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re

def tokenize(inst):
    lst = inst.split()
    return lst

def better_tokenize(inst):
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

def train(alpha,tokenize):
    ny0=0
    ny1=0
    dct_nxi_y0 = dict()
    dct_nxi_y1 = dict()
    dct_p_xi_y0 = dict()
    dct_p_xi_y1 = dict()

    with open('train.tsv') as tr:
        reader = csv.DictReader(tr, dialect='excel-tab')
        for row in reader:
            lst_word = tokenize(row['text'])
            if row['class'] == '0':
                for w in lst_word:
                    dct_nxi_y0[w] = dct_nxi_y0.get(w,0)+1
                ny0 += 1
            else:
                for w in lst_word:
                    dct_nxi_y1[w] = dct_nxi_y1.get(w,0)+1
                ny1 += 1

    ny = ny0+ny1
    p_y0_y = ny0/ny
    p_y1_y = ny1/ny

    # Vx_in_y0 = len(dct_nxi_y0)
    # Vx_in_y1 = len(dct_nxi_y1)

    nx_in_y0 = sum(dct_nxi_y0.values())
    nx_in_y1 = sum(dct_nxi_y1.values())

    x_st = set(dct_nxi_y0.keys())|set(dct_nxi_y1.keys())
    Vx_in_y = len(x_st)

    for xi in x_st:
        dct_p_xi_y0[xi] = (dct_nxi_y0.get(xi,0) + alpha)/(nx_in_y0 + Vx_in_y * alpha)
        dct_p_xi_y1[xi] = (dct_nxi_y1.get(xi,0) + alpha)/(nx_in_y1 + Vx_in_y * alpha)

    dct_p = {'p_y0_y':p_y0_y,'p_y1_y':p_y1_y,'p_xi_y0':dct_p_xi_y0,'p_xi_y1':dct_p_xi_y1}

    return dct_p

def classify(token,dct_p):
    p_y0_y = dct_p['p_y0_y']
    p_y1_y = dct_p['p_y1_y']
    dct_p_xi_y0 = dct_p['p_xi_y0']
    dct_p_xi_y1 = dct_p['p_xi_y1']

    p_x_y0 = 1
    p_x_y1 = 1
    for xi in token:
        if xi not in dct_p_xi_y0.keys():
            continue
        p_x_y0 = p_x_y0 * dct_p_xi_y0[xi]
        p_x_y1 = p_x_y1 * dct_p_xi_y1[xi]

    if p_x_y0 + p_x_y1 == 0: #for example, doc1 = +:a,b; doc2 = -:c,d; then p(a,c|+)=p(a,c|+)=0
        p_y0_x = 0
        p_y1_x = 0
    else:
        p_y0_x = p_x_y0 * p_y0_y/(p_x_y0 * p_y0_y + p_x_y1 * p_y1_y)
        p_y1_x = p_x_y1 * p_y1_y/(p_x_y0 * p_y0_y + p_x_y1 * p_y1_y)

    if p_y0_x > p_y1_x:
        return 0
    else:
        return 1


y_true = list()
y_pred = list()

def f1_generation(a,tokenize):
    with open('dev.tsv') as dev:
        reader2 = csv.DictReader(dev, dialect='excel-tab')
        dct_p = train(a,tokenize)
        for row in reader2:
            lst_txt = tokenize(row['text'])
            y_pred.append(classify(lst_txt,dct_p))
            y_true.append(int(row['class']))

    F1 = f1_score(y_true, y_pred,average = 'micro')
    return F1

f1_1_t = f1_generation(1,tokenize)
f1_1_b = f1_generation(1,better_tokenize)
print ('tokenzie: ',f1_1_t)
print ('better_tokenzie: ', f1_1_b)

lst_a= [0,0.2,0.4,0.6,0.8,1,1.2,1.4]
lst_f1 = list()
for a in lst_a:
    lst_f1.append(f1_generation(a,tokenize))
plt.plot(lst_a,lst_f1)
plt.show()
