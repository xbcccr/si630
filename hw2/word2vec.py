import os,sys,re,csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from numba import jit
import matplotlib.pyplot as plt



#... (1) First load in the data source and tokenize into one-hot vectors.
#... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


#... (2) Prepare a negative sampling distribution table to draw negative samples from.
#... Consistent with the original word2vec paper, this distribution should be exponentiated.


#... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
#... This training will occur through backpropagation from the context words down to the source word.


#... (4) Re-train the algorithm using different context windows. See what effect this has on your results.


#... (5) Test your model. Compare cosine similarities between learned word vectors.










#.................................................................................
#... global variables
#.................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10


vocab_size = 0
hidden_size = 100
uniqueWords = [""]                      #... list of all unique tokens
wordcodes = {}                          #... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()                  #... how many times each token occurs
samplingTable = []                      #... table to draw negative samples from

unk_idx = 0
samplingTable_last_key = 0



#.................................................................................
#... load in the data and convert tokens to one-hot indices
#.................................................................................
def debug(x,des):
	print ('debugging', str(des), " is: ", x)

def loadData(filename):
	global uniqueWords, wordcodes, wordcounts, unk_idx
	override = True
	if override:
		#... for debugging purposes, reloading input file and tokenizing is quite slow
		#...  >> simply reload the completed objects. Instantaneous.
		fullrec = pickle.load(open("w2v_fullrec.p","rb"))
		wordcodes = pickle.load( open("w2v_wordcodes.p","rb"))
		uniqueWords= pickle.load(open("w2v_uniqueWords.p","rb"))
		wordcounts = pickle.load(open("w2v_wordcounts.p","rb"))
		return fullrec


	#... load in the unlabeled data file. You can load in a subset for debugging purposes.
	handle = open(filename, "r", encoding="utf8")
	fullconts =handle.read().split("\n")
	fullconts = [line.split("\t")[1].replace("<br />", "") for line in fullconts[1:(len(fullconts)-1)]]
	#now fullcounts is a list of text of each line

	#... apply simple tokenization (whitespace and lowercase)
	fullconts = [" ".join(fullconts).lower()] #a list of one element
	debug(fullconts[0][:50],'fullconts[0][:50]')





	print ("Generating token stream...")
	#... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
	#... for simplicity, you may use nltk.word_tokenize() to split fullconts.
	#... keep track of the frequency counts of tokens in origcounts.
	min_count = 50
	origcounts = Counter()
	lst_stopwords = set(stopwords.words('english'))
	lst_words = nltk.word_tokenize(fullconts[0])
	fullrec = [token for token in lst_words if token not in lst_stopwords]
	origcounts.update(fullrec)
	debug(fullrec[:10],'fullrec[:10]')
	# debug([(k,origcounts[k]) for k in list(origcounts.keys())[:10]], 'origcounts[:10]')



	print ("Performing minimum thresholding..")
	#... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
	#... replace other terms with <UNK> token.
	fullrec_filtered = []
	for token in fullrec:
		if origcounts[token] < min_count:
			token = '<UNK>'
		fullrec_filtered.append(token)
	debug(fullrec_filtered[:50],'fullrec_filtered[:50]')



	#... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)
	wordcounts.update(fullrec_filtered)
	# debug([(k,wordcounts[k]) for k in list(wordcounts.keys())[:10]],'wordcounts[:10]')
	#... after filling in fullrec_filtered, replace the original fullrec with this one.
	fullrec = fullrec_filtered





	print ("Producing one-hot indicies")
	#... (TASK) sort the unique tokens into array uniqueWords
	#... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
	uniqueWords = sorted(list(wordcounts.keys()))


	for i in range(len(uniqueWords)):
		token = uniqueWords[i]
		onehot_index = i
		if token == '<UNK>':
			unk_idx = i
		wordcodes[token] = onehot_index
	debug(unk_idx,'unk_idx')
	# debug([(k,v) for k,v in wordcodes.items()[:50]],'wordcodes[:50]')
	#... replace all word tokens in fullrec with their corresponding one-hot indices.
	for i in range(len(fullrec)):
		fullrec[i] = wordcodes[fullrec[i]]
	debug(fullrec[:50],'fullrec[:50]')






	#... close input file handle
	handle.close()



	#... store these objects for later.
	#... for debugging, don't keep re-tokenizing same data in same way.
	#... just reload the already-processed input data with pickles.
	#... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

	pickle.dump(fullrec, open("w2v_fullrec.p","wb+"))
	pickle.dump(wordcodes, open("w2v_wordcodes.p","wb+"))
	pickle.dump(uniqueWords, open("w2v_uniqueWords.p","wb+"))
	pickle.dump(dict(wordcounts), open("w2v_wordcounts.p","wb+"))


	#... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
	return fullrec







#.................................................................................
#... compute sigmoid value
#.................................................................................
@jit(nopython=True)
def sigmoid(x):
	return float(1)/(1+np.exp(-x))









#.................................................................................
#... generate a table of cumulative distribution of words
#.................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power=0.75):
	global samplingTable_last_key
	#... stores the normalizing denominator (count of all tokens, each count raised to exp_power)
	max_exp_count = 0



	print ("Generating exponentiated count vectors")
	#... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
	#... store results in exp_count_array.
	exp_count_array = []
	for token in uniqueWords:
		exp_count_array.append(np.power(wordcounts[token],exp_power))
	# debug(exp_count_array[:50],'exp_count_array[:50]')

	max_exp_count = sum(exp_count_array)
	# debug(max_exp_count,'max_exp_count')


	print ("Generating distribution")

	#... (TASK) compute the normalized probabilities of each term.
	#... using exp_count_array, normalize each value by the total value max_exp_count so that
	#... they all add up to 1. Store this corresponding array in prob_dist
	prob_dist = np.divide(exp_count_array,max_exp_count) #get an ndarray
	# debug(prob_dist[:50],'prob_dist[:50]')




	print ("Filling up sampling table")
	#... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
	#... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
	#... multiplied by table_size. This table should be stored in cumulative_dict.
	#... we do this for much faster lookup later on when sampling from this table.

	cumulative_dict = dict()
	global table_size
	table_size = 1e8
	key = 0
	# debug(len(prob_dist),'should be same with len(uniqueWords):11219')
	for i in range(len(prob_dist)):
		prob = prob_dist[i]
		size = prob * table_size
		size = int(round(size))
		for count in range(size):
			# cumulative_dict[key] = wordcodes[uniqueWords[i]]
			cumulative_dict[key] = i
			key += 1
	# debug(size,'last size of the last word')
	# debug(list(cumulative_dict.keys())[:10])
	samplingTable_last_key = key-1
	debug(key-1,'last key')
	debug(uniqueWords[cumulative_dict[key-1]],'last word')
	debug(uniqueWords[-1],'last word')
	return cumulative_dict






#.................................................................................
#... generate a specific number of negative samples
#.................................................................................


def generateSamples(context_idx, num_samples):
	global samplingTable, uniqueWords,samplingTable_last_key
	results = []
	#... (TASK) randomly sample num_samples token indices from samplingTable.
	#... don't allow the chosen token to be context_idx.
	#... append the chosen indices to results
	i = 0
	for i in range(num_samples):
		key = random.randint(0,samplingTable_last_key) #randint(a,b)= a int in [a,b]
		ng_idx = samplingTable[key]
		while ng_idx == context_idx:
			key = random.randint(0,table_size-1)
			ng_idx = samplingTable[key]
		results.append(ng_idx)

	return results









@jit(nopython=True)
def performDescent(num_samples, learning_rate, center_token_idx, sequence_chars,W1,W2,negative_indices):
	# sequence chars = mapped_context = indices for context word
	# center_token = index for center word
	# hidden_size = 100
	nll_new = 0
	#lst_j will be a list of indices of context_token and its negative samples

	for i in range(0, len(sequence_chars)):
		#...Run gradient descent on both
		#... weight matrices W1 and W2.
		#... compute the total negative log-likelihood and store this in nll_new.

		#key is the current context token from sequence_chars
		#find the associated negative samples from negative_indices.
		#ng_oh_idx refers to indices of nagative smapling for one context word
		c_idx = sequence_chars[i]
		lst_j = []
		lst_j.append(c_idx)
		for ii in range(num_samples):
			# print (num_samples,k,i, negative_indices)
			ng_idx = negative_indices[num_samples * i + ii]
			lst_j.append(ng_idx)
			ii += 1
		# print ('lst_j: (shold be 3, 1 for context, 2 for ng)',lst_j)

		#if you store the old vector value in a variable, make sure you are storing a copy of that vector by using,
		#for example, np.copy(W2[...]) instead of just W2[...].
		#Otherwise, updates to W2 will automatically update the old value as well, and vice visa
		h_copy = np.copy(W1[center_token_idx])
		vi_copy =  np.copy(W1[center_token_idx])#vi is word embedding for input word
		for j in lst_j:
			v2_j_copy = np.copy(W2[j])
			if j == c_idx:
				tj = 1
			else:
				tj = 0

			#update vi, and as it's not a copy, it updates W1 meanwhile (but it needs a full loop of lst_j to finish update of W1)
			vi_copy = vi_copy - learning_rate * (sigmoid(np.dot(v2_j_copy,h_copy))-tj) * v2_j_copy
			#update W2, and as it's not a copy, it updates W2 meanwhile
			v2_j_copy = v2_j_copy - learning_rate * (sigmoid(np.dot(v2_j_copy,h_copy))-tj) * h_copy
			#update W2
			W2[j] = v2_j_copy

		# update h to the newest
		W1[center_token_idx] = vi_copy
		h = vi_copy
		#caculating nll for a context word and its negtive samples
		for j in lst_j:
			if j == c_idx:
				tj = 1
			else:
				tj = 0
			v2_j = W2[j]

			# debug(tj,'tj')
			# debug(np.dot(v2_j,h.T),'np.dot(v2_j,h.T')
			# debug(sigmoid((-1)**(1-tj)*np.dot(v2_j,h.T)),'sigmoid((-1)**(1-tj)*np.dot(v2_j,h.T))')
			# debug(- np.log(sigmoid((-1)**(1-tj)*np.dot(v2_j,h.T))),'- np.log(sigmoid((-1)**(1-tj)*np.dot(v2_j,h.T)))')

#numba.errors.TypingError: Failed at nopython (nopython frontend)
#cannot unify int64 and array(float64, 2d, C) for 'nll_new', defined at word2vec.py (301)
			nll_new += - np.log(sigmoid((-1)**(1-tj)*np.dot(v2_j,h))) # a positive number






	# print('one performDescent done')
	# print()
	return nll_new






#.................................................................................
#... learn the weights for the input-hidden and hidden-output matrices
#.................................................................................


def trainer(curW1 = None, curW2=None):
	global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size,np_randcounter, randcounter, unk_idx
	vocab_size = len(uniqueWords)           #... unique characters
	hidden_size = 100                       #... number of hidden neurons
	context_window = [-2,-1,1,2]            #... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
	nll_results = []
	unk_idx = wordcodes['<UNK>']                      					#... keep array of negative log-likelihood after every 1000 iterations
	# debug(unk_idx,'unk_idx')

	#... determine how much of the full sequence we can use while still accommodating the context window
	start_point = int(math.fabs(min(context_window)))
	end_point = len(fullsequence)-(max(max(context_window),0))
	mapped_sequence = fullsequence



	#... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
	if curW1==None:
		np_randcounter += 1
		W1 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
		W2 = np.random.uniform(-.5, .5, size=(vocab_size, hidden_size))
	else:
		#... initialized from pre-loaded file
		W1 = curW1
		W2 = curW2



	#... set the training parameters
	epochs = 1
	num_samples = 2
	learning_rate = 0.05
	nll = 0
	iternum = 0




	#... Begin actual training
	for j in range(0,epochs):
		print ("Epoch: ", j)
		prevmark = 0

		#... For each epoch, redo the whole sequence...
		for i in range(start_point,end_point):
			#i is the sequence of token in the observation, not onehot_index from uniqueWords

			if (float(i)/len(mapped_sequence))>=(prevmark+0.1):
				print ("Progress: ", round(prevmark+0.1,1))
				prevmark += 0.1
			if iternum%10000==0:
				print ("Negative likelihood: ", nll)
				nll_results.append(nll)
				nll = 0


			#... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
			center_token_idx = mapped_sequence[i] #onehot_index
			# debug(uniqueWords[center_token_idx],'center_token')
			# debug(center_token_idx,'center token index')
			# debug(unk_idx,'unk_idx')
			#... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
			if center_token_idx == unk_idx:
				# print()
				# print('unk,will skip')
				# print()
				continue




			iternum += 1
			#... now propagate to each of the context outputs
			mapped_context = [mapped_sequence[i+ctx] for ctx in context_window]
			# debug(mapped_context,'context_idx')
			negative_indices = []
			for q in mapped_context:
				# debug(q,'one context')
				negative_indices += generateSamples(q, num_samples)
			nll_new = performDescent(num_samples, learning_rate, center_token_idx, mapped_context, W1,W2, negative_indices)
			nll += nll_new



	lst_steps = []
	iii=0
	for nll_res in nll_results:
		lst_steps.append(10000*iii)
		iii += 1
	plt.plot(lst_steps,nll_results)
	plt.savefig('steps_nll.png')
	plt.close()

	return [W1,W2]



#.................................................................................
#... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
#.................................................................................

def load_model():
	handle = open("saved_W1.data(4)","rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data(4)","rb")
	W2 = np.load(handle)
	handle.close()
	return [W1,W2]






#.................................................................................
#... Save the current results to an output file. Useful when computation is taking a long time.
#.................................................................................

def save_model(W1,W2):
	handle = open("saved_W1.data","wb+")
	np.save(handle, W1, allow_pickle=False)
	handle.close()

	handle = open("saved_W2.data","wb+")
	np.save(handle, W2, allow_pickle=False)
	handle.close()






#... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
#... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
#... vector predict similarity to a context word.






#.................................................................................
#... code to start up the training function.
#.................................................................................
word_embeddings = []
proj_embeddings = []
def train_vectors(preload=False):
	global word_embeddings, proj_embeddings
	if preload:
		[curW1, curW2] = load_model()
	else:
		curW1 = None
		curW2 = None
	[word_embeddings, proj_embeddings] = trainer(curW1,curW2)
	save_model(word_embeddings, proj_embeddings)









#.................................................................................
#... for the averaged morphological vector combo, estimate the new form of the target word
#.................................................................................

def morphology(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [word_seq[0], # suffix averaged
	embeddings[wordcodes[word_seq[1]]]]
	vector_math = vectors[0]+vectors[1]
	#... find whichever vector is closest to vector_math
	#... (TASK) Use the same approach you used in function prediction() to construct a list
	#... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
#.................................................................................

def analogy(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [embeddings[wordcodes[word_seq[0]]],
	embeddings[wordcodes[word_seq[1]]],
	embeddings[wordcodes[word_seq[2]]]]
	vector_math = -vectors[0] + vectors[1] - vectors[2] # + vectors[3] = 0
	#... find whichever vector is closest to vector_math
	#... (TASK) Use the same approach you used in function prediction() to construct a list
	#... of top 10 most similar words to vector_math. Return this list.







#.................................................................................
#... find top 10 most similar words to a target word
#.................................................................................


def prediction(target_word):
	global word_embeddings, uniqueWords, wordcodes
	targets = [target_word]
	outputs = []
	#... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
	#... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
	#... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
	#... return a list of top 10 most similar words in the form of dicts,
	#... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}











if __name__ == '__main__':
	if len(sys.argv)==2:
		filename = sys.argv[1]
		#... load in the file, tokenize it and assign each token an index.
		#... the full sequence of characters is encoded in terms of their one-hot positions

		fullsequence= loadData(filename)
		print ("Full sequence loaded...")
		#print(uniqueWords)
		#print (len(uniqueWords))



		#... now generate the negative sampling table
		print ("Total unique words: ", len(uniqueWords))
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)
		samplingTable_keys = list(samplingTable.keys())


		#... we've got the word indices and the sampling table. Begin the training.
		#... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
		#... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
		#... ... and uncomment the load_model() line
		print()
		print('begin to train')
		train_vectors(preload=False)
		# [word_embeddings, proj_embeddings] = load_model()








		#... we've got the trained weight matrices. Now we can do some predictions
		targets = ["good", "bad", "scary", "funny"]
		for targ in targets:
			print("Target: ", targ)
			bestpreds= (prediction(targ))
			for pred in bestpreds:
				print (pred["word"],":",pred["score"])
			print ("\n")



		#... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
		print (analogy(["son", "daughter", "man"]))
		print (analogy(["thousand", "thousands", "hundred"]))
		print (analogy(["amusing", "fun", "scary"]))
		print (analogy(["terrible", "bad", "amazing"]))



		#... try morphological task. Input is averages of vector combinations that use some morphological change.
		#... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
		#... the morphology() function.

		s_suffix = [word_embeddings[wordcodes["stars"]] - word_embeddings[wordcodes["star"]]]
		others = [["types", "type"],
		["ships", "ship"],
		["values", "value"],
		["walls", "wall"],
		["spoilers", "spoiler"]]
		for rec in others:
			s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
		s_suffix = np.mean(s_suffix, axis=0)
		print (morphology([s_suffix, "techniques"]))
		print (morphology([s_suffix, "sons"]))
		print (morphology([s_suffix, "secrets"]))






	else:
		print ("Please provide a valid input filename")
		sys.exit()
