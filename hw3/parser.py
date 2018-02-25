import sys
import math
import numpy as np
from nltk.tree import *
from numbers import Number


def load_grammar(filename):
	'''
	load grammar that following CNF
	return a dict = {(A, (B, C)): prob, (A, a): prob}
	A, B and C represent non-terminal rules and a is lexicon
	prob is  normalized probability
	'''
	# ...(TASK) load grammar and return normalized probability for each production rule
	grammarProb = {}
	trackweights = {}

	with open(filename,'r') as handle:
		for line in handle:
			line = line.rstrip()
			line = line.split()
			if len(line) == 0 or line[0] == '#':
				continue
			if len(line) in [3,4]:
				weight = float(line[0])
				A = line[1]
				trackweights[A] = trackweights.get(A,0) + weight
				if len(line) == 3:
					terminal = line[2]
					grammarProb[(A,terminal)] = weight
				else:
					non_terminal_1 = line[2]
					non_terminal_2 = line[3]
					grammarProb[(A,(non_terminal_1,non_terminal_2))] = weight

	for rule in grammarProb:
		A = rule[0]
		grammarProb[rule] = grammarProb[rule]/trackweights[A]
		# print(rule,grammarProb[rule])

	return grammarProb


def parse(words, grammar):

	sentenceLen = len(words)

	# ...initialize score table and backpointer table
	score = [[{} for i in range(sentenceLen+1)] for j in range(sentenceLen)]
	backpointer = [[{} for i in range(sentenceLen+1)] for j in range(sentenceLen)]


	# ...(TASK) fill up score and backpointer table
	print ('span = 1')
	for i in range(sentenceLen):
		word = words[i]
		for rule, prob in grammar.items():
			if rule[1] == word:
				score[i][i+1][rule[0]] = prob


	for span in range(2,sentenceLen+1):
		print('spen=',span)
		for i in range(sentenceLen - span + 1):
			begin = i
			end = begin + span
			print((begin,end))
			for split in range(begin+1,end):
				#B in cell (begin,split),C in cell (split,end)
				print(split)
				for rule, prob in grammar.items():
					if type(rule[1]) is tuple:
						# print('hh')
						# print(rule[1])
						A = rule[0]
						B = rule[1][0]
						C = rule[1][1]
						if B in score[begin][split] and C in score[split][end]:
							#in case there's already rule for A, need to compare here:
							if A in score[begin][end]:
								A_prob_ini = score[begin][end][A]
								A_prob_now = prob * score[begin][split][B] * score[split][end][C]
								if A_prob_ini < A_prob_now:
									score[begin][end][A] = A_prob_now
									backpointer[begin][end][A] = (split,B,C)
							else:
								score[begin][end][A] = prob * score[begin][split][B] * score[split][end][C]
								backpointer[begin][end][A] = (split,B,C)
	# ...(TASK) return flag invalidParse and max probability parser can get
	if len(score[0][sentenceLen]) == 0:
		invalidParse = True
		maxScore = 0
	else:
		invalidParse = False
		maxScore = max(score[0][sentenceLen].values())

	print(score)
	print(backpointer)
	return invalidParse, maxScore, backpointer






#... A => B,C, arr1 is for B and arr2 is for C
def addBranch(words, backpointer, arr1, arr2):

	[start1, end1, symb1] = arr1
	[start2, end2, symb2] = arr2

	# for first non-terminal/terminal
	if (end1-start1==1):
		tree1 = Tree(symb1,[words[start1]])
	else:
		B = backpointer[start1][end1][symb1]


		split, R1,R2 = B
		split1a = [start1, split]
		split1b = [split, end1]

		tree1 = Tree(symb1, addBranch(words, backpointer, [start1, split, R1], [split, end1, R2]))


	# for second non-terminal/terminal
	if (end2-start2==1):
		tree2 = Tree(symb2,[words[start2]])
	else:
		C = backpointer[start2][end2][symb2]
		split, R1,R2 = C
		split1a = [start2, split]
		split1b = [split, end2]

		tree2 = Tree(symb2, addBranch(words, backpointer, [start2, split, R1], [split, end2, R2]))

	return [tree1, tree2]





def pretty_print(words, backpointer):

	#... start at the root of the tree
	foundRoot = False
	sentLen = len(backpointer)
	for key,value in backpointer[0][-1].items():
		if key=="S": #... this is the root, REQUIRED symbol
			foundRoot = True
			split, B,C = value
			tree = Tree(key, addBranch(words, backpointer, [0,split,B], [split,sentLen,C]))
			break


	if foundRoot:
		tree.pretty_print()
	else:
		#... This grammar could not match the provided sentence.
		print ("Cannot find root")
		return

	return tree



def main():
	if len(sys.argv) != 4:
		print(('Wrong number of arguments?: %d\nExpected python parser.py ' +
			   'grammar.gr lexicon.txt sentences.txt') % (len(sys.argv)-1))
		exit(1)

	grammar_file = sys.argv[1]
	lexicon_file = sys.argv[2]
	sentences_file = sys.argv[3]


	#... we're assuming that lexicon.txt is line-separated with each line containing
	#... exactly one token that is permissible. The rules for these tokens is contained
	#... in grammar.gr
	lexicon = set()
	with open(lexicon_file) as f:
		for line in f:
			lexicon.add(line.strip())
	print("Saw %d terminal symbols in the lexicon" % (len(lexicon)))


	grammar = load_grammar(grammar_file)


	# non_terminals = get_non_terminals(grammar, lexicon)

	with open(sentences_file) as f:
		for line in f:
			words = line.strip().split()
			invalidParse, score, backpointer = parse(words, grammar)
			if invalidParse:
				print ("Grammar couldn't parse this sentence")
			else:
				print('%f\t%s' % (score, pretty_print(words,backpointer)))


if __name__ == '__main__':
	main()
