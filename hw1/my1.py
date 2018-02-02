import csv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import numpy as np


def tokenize(inst):
    lst = inst.split()
    return lst

# def better_tokenize(inst):
#     line = inst.lower()
#     lst_1 = line.split()
#     lst_2 = []#lst_1.copy()
#
#     for w in lst_1:
#         p1 = r'[a-z]+'
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
def better_tokenize(inst):
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

def train(tokenize, alpha=0):
    ny0=0
    ny1=0
    dct_nxi_y0 = dict()
    dct_nxi_y1 = dict()
    dct_p_xi_y0 = dict()
    dct_p_xi_y1 = dict()

    with open('train.tsv','r', encoding= 'utf-8') as f:
        rdata = []
        reader = f.readlines()[1:]
        for line in reader:
            rdata.append(re.split("\t",line.replace('\n','')))
        for row in rdata:
            lst_word = tokenize(row[1])
            if row[2] == '0':
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

def predict(file,tokenize,with_label, a=0):
    y_true = list()
    y_pred = list()
    y_id = list()
    with open(file,'r', encoding= 'utf-8') as f:
        rdata = []
        reader = f.readlines()[1:]
        for line in reader:
            rdata.append(re.split("\t",line.replace('\n','')))

        dct_p = train(tokenize,a)
        for row in rdata:
            lst_txt = tokenize(row[1])
            y_pred.append(classify(lst_txt,dct_p))
            y_id.append(row[0])
            if with_label:
                y_true.append(int(row[2]))
    return {'y_true':y_true,'y_pred':y_pred,'y_id':y_id}

def f1(y_true, y_pred,average = 'micro'):
    F1 = f1_score(y_true, y_pred, average = average)
    return F1

def f1_short(file,tokenize,with_label, a=0,average = 'micro'):
    dct = predict(file,tokenize,with_label,a)
    f1_score = f1(dct['y_true'],dct['y_pred'],average = average)
    return f1_score


dct = predict('dev.tsv',tokenize,True)
f1_a0_t = f1(dct['y_true'],dct['y_pred'])
print ('no smoothing, a=0: ',f1_a0_t)


#check a's influence on f1
lst_a= list(np.arange(0,2,0.1))
lst_f1 = list()
f_max = -100
for a in lst_a:
    dct = predict('dev.tsv',better_tokenize,True,a)
    F1= f1(dct['y_true'],dct['y_pred'])
    lst_f1.append(F1)
    if f_max < F1:
        best_a = a
        f_max = F1

plt.plot(lst_a,lst_f1) #when a =0.6, f1 peaks
plt.xlabel('smoothing value')
plt.ylabel('f1 score')
plt.title(best_a)
plt.savefig('f1-a.png')

#check better better_tokenize
f1_a1_t = f1_short('dev.tsv',tokenize,True,1)
f1_a1_b = f1_short('dev.tsv',better_tokenize,True,1)
print ('tokenize, a = 1: ',f1_a1_t)
print ('better_tokenzie, a =1: ', f1_a1_b)

# write a csv to kaggle
dct = predict('test.unlabeled.tsv',better_tokenize, False, a=0.6)
y_pred = dct['y_pred']
y_id = dct['y_id']

with open('test1.csv', 'w') as test1:
    test1.write("instance_id,class\n")
    for i in range(len(y_pred)):
        test1.write(y_id[i]+","+str(y_pred[i])+"\n")
