from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
import math

def Read_data():
	corpus = loadmat("data/corpus.mat")
	corpus_objects = []
	corpus_words = []
	for i in range(len(corpus["corpus"][0])):
		corpus_objects += [corpus["corpus"][0][i][0][0]]
		corpus_words += [corpus["corpus"][0][i][1][0]]
	corpus_objects = np.array(corpus_objects)
	corpus_words = np.array(corpus_words)

	world = loadmat("data/world.mat")
	world_words = world["world"][0][0][1][0]
	world_objects = world["world"][0][0][4][0]
	word_freq = world["world"][0][0][6][0]
	object_freq = world["world"][0][0][7][0]
	world_mut = world["world"][0][0][8]
	world_mut = torch.from_numpy(world_mut)

	# for i in range(len(corpus_objects)):
	# 	print("----------%s---------" %(i))
	# 	print(corpus_words[i])
	# 	print(world_words[corpus_words[i] - 1])
	# 	print(corpus_objects[i])
	# 	for each in corpus_objects[i]:
	# 		print(world_objects[each - 1])

	gold_standard = loadmat("data/gold_standard.mat")
	gold_standard_word = gold_standard["gold_standard"][0][0][1][0]
	gold_standard_object = gold_standard["gold_standard"][0][0][1][1]

	# print(world_words[-1])
	# print(world_objects)
	# print("Ground Truth:\n")
	# for i in range(len(gold_standard_word)):
	# 	print("%s   %s" %(world_words[gold_standard_word[i] - 1], world_objects[gold_standard_object[i] - 1]))

	return corpus_objects, corpus_words, world_words, world_objects, word_freq, object_freq, gold_standard_word, gold_standard_object, world_mut

def Get_intention_object(obj_num, gamma):
	mat = torch.zeros((2**obj_num, obj_num), dtype = torch.float64)
	for i in range(2**obj_num):
		curr = i
		for j in range(obj_num):
			mat[i, obj_num - j - 1] = curr%2
			curr = curr // 2

	mat = (mat.T / (torch.sum(mat, dim = 1).T + 1e-10)).T * gamma

	extra = torch.ones((2**obj_num, 1), dtype = torch.float64)
	for i in range(2 ** obj_num):
		extra[i, 0] = 1 - gamma
	extra[0, 0] = 1.0
	mat = torch.cat((extra, mat), dim = 1)
	return mat

def Initialize_lexicon(num_temp, world_words, world_objects):
	result = []
	num_words = len(world_words)
	num_objects = len(world_objects)
	for i in range(num_temp):
		num_pairs = torch.randint(1,10,[1])
		result_temp = torch.zeros(num_pairs, 2).long()
		result_temp[:,0] = torch.randint(0, num_words, [num_pairs])
		result_temp[:,1] = torch.randint(0, num_objects, [num_pairs])
		result += [result_temp]
	return result

def Get_object_words(lexicon, corpus_objects, corpus_words, kappa, num_words, num_objects):
	mat = torch.zeros((num_objects+1, num_words), dtype = torch.float64)
	mat[0] = 1
	
	#referential, non-referential
	obj_dict = dict()
	for i in range(lexicon.shape[0]):
		if obj_dict.__contains__(lexicon[i, 1]):
			obj_dict[lexicon[i, 1].item()] += [lexicon[i, 0].item()]
		else:
			obj_dict[lexicon[i, 1].item()] = [lexicon[i, 0].item()]
	for key, value in obj_dict.items():
		for each_value in value:
			mat[key+1, each_value] = 1 / len(value)
			# non-referential, divided by kappa
			if mat[0, each_value] == 1:
				mat[0, each_value] = kappa

	mat[0] = mat[0] / torch.sum(mat[0])

	for i in range(num_objects):
		if i not in obj_dict.keys():
			mat[i+1] = mat[0]

	return mat


def Evaluate_lexicon(lexicon, corpus_objects, corpus_words, alpha, num_words, num_objects, lexicon_mat, intention_mats):
	prob_prior_log = (-alpha * lexicon.shape[0])

	# Initialize corpus probability
	prob_corpus_log = torch.zeros(len(corpus_objects))

	for i in range(len(corpus_objects)):
		curr_situ_words = (corpus_words[i]-1).tolist()
		curr_situ_objects = corpus_objects[i].tolist()
		curr_situ_objects = [0] + curr_situ_objects
		curr_situ_mat = lexicon_mat[curr_situ_objects]
		curr_situ_mat = curr_situ_mat[:, curr_situ_words]
		curr_situ_mat = torch.matmul(intention_mats[i], curr_situ_mat)
		curr_situ_mat = torch.prod(curr_situ_mat, dim = 1) # Multiply over words to get prob for sentences on particular intention
		curr_situ_mat = torch.mean(curr_situ_mat, dim = 0)*(math.e**50) # Average over intentions to get prob for sentences
		prob_corpus_log[i] = torch.log(curr_situ_mat) - 50
		if prob_corpus_log[i] < -1000:
			print(i)
			print("Warning: too low probability")

	lexicon_prob = torch.sum(prob_corpus_log) + prob_prior_log # Add over all corpus and prior prob to get lexicon prob

	return lexicon_prob


def Mutate(curr_lexicon, world_mut):
	# Make a random change
	change = torch.randint(0,3,[1])
	if change == 0:
		return Add_pair(curr_lexicon, world_mut)
	if change == 1:
		return Del_pair(curr_lexicon)
	if change == 2:
		return Swap_pair(curr_lexicon, world_mut)

	print("Check 'change' value.")
	return None


def Add_pair(curr_lexicon, world_mut):
	flap_mut = (world_mut.reshape(-1) / torch.sum(world_mut))
	index = np.random.choice(np.arange(0,flap_mut.shape[0]), p = flap_mut.numpy())
	index_x = index // world_mut.shape[1]
	index_y = index % world_mut.shape[1]
	extra = torch.tensor([index_x, index_y]).reshape(1,2)
	result = torch.cat((curr_lexicon, extra), dim = 0)
	return result, index_x, index_y, "add"

def Del_pair(curr_lexicon):
	result = curr_lexicon
	if curr_lexicon.shape[0] > 0:
		delete_idx = torch.randint(0, curr_lexicon.shape[0], [1])
		index_x = curr_lexicon[delete_idx, 0]
		index_y = curr_lexicon[delete_idx, 1]
		result = torch.cat((curr_lexicon[0:delete_idx], curr_lexicon[delete_idx+1:]), dim = 0)
		return result, index_x.item(), index_y.item(), "delete"
	return result, 0, 0, "no change"

def Swap_pair(curr_lexicon, world_mut):
	flap_mut = (world_mut.reshape(-1) / torch.sum(world_mut))
	index = np.random.choice(np.arange(0,flap_mut.shape[0]), p = flap_mut.numpy())
	index_x = index // world_mut.shape[1]
	index_y = index % world_mut.shape[1]
	candidate = (curr_lexicon[:,1] == index_y).nonzero()
	if candidate.shape[0] == 0:
		extra = torch.tensor([index_x, index_y]).reshape(1,2)
		result = torch.cat((curr_lexicon, extra), dim = 0)
		return result, index_x, index_y, "add"
	else:
		replace_index = candidate[torch.randint(0, candidate.shape[0], [1]), 0]
		result = curr_lexicon.clone()
		result[replace_index, 0] = index_x.item()
		return result, index_x, index_y, "swap"

def Breed(lexicon_a, lexicon_b, temperature_a, temperature_b):
	# Make a random breed
	option = torch.randint(0,5,[1])
	if option == 0:
		return Site_borrow(lexicon_a, lexicon_b)
	if option == 1:
		return Word_swap(lexicon_a, lexicon_b)
	if option == 2:
		return Object_swap(lexicon_a, lexicon_b)
	if option == 3:
		return Site_swap(lexicon_a, lexicon_b)
	if option == 4:
		if temperature_a < temperature_b or torch.rand(1)<0.1:
			return Swap(lexicon_a, lexicon_b)
		else:
			return lexicon_a
	print("Check 'option' value.")
	return None

def Site_borrow(lex_a, lex_b):
	result = lex_a.clone()
	if lex_b.shape[0] != 0:
		add_idx = torch.randint(0,lex_b.shape[0],[1])
		extra = lex_b[add_idx]
		result = torch.cat((lex_a, extra), dim = 0)
	return result

def Word_swap(lex_a, lex_b):
	result = lex_a.clone()
	if lex_a.shape[0] != 0:
		replace_word_idx = torch.randint(0, result.shape[0], [1])
		replace_word = result[replace_word_idx, 0]
		remain_idx = (result[:,0] != replace_word).nonzero()
		result = result[remain_idx].reshape(-1,2)
		b_add_idx = (lex_b[:,0] == replace_word).nonzero()
		result = torch.cat((result, lex_b[b_add_idx].reshape(-1,2)), dim = 0)
	return result

def Object_swap(lex_a, lex_b):
	result = lex_a.clone()
	if lex_a.shape[0] != 0:
		replace_object_idx = torch.randint(0, result.shape[0], [1])
		replace_object = result[replace_object_idx, 1]
		remain_idx = (result[:, 1] != replace_object).nonzero()
		result = result[remain_idx].reshape(-1, 2)
		b_add_idx = (lex_b[:, 1] == replace_object).nonzero()
		result = torch.cat((result, lex_b[b_add_idx].reshape(-1,2)), dim = 0)
	return result

def Site_swap(lex_a, lex_b):
	result = lex_a.clone()
	if lex_a.shape[0] != 0 and lex_b.shape[0] != 0:
		swapb_idx = torch.randint(0, lex_b.shape[0], [1])
		swapa_idx = torch.randint(0, lex_a.shape[0], [1])
		result[swapa_idx] = lex_b[swapb_idx]
	return result

def Swap(lex_a, lex_b):
	return lex_b.clone()

if __name__ == "__main__":
	torch.manual_seed(1)
	np.random.seed(1)
	print("Random seed: 1")
	corpus_objects, corpus_words, world_words, world_objects, word_freq, object_freq, gold_standard_word, gold_standard_object, world_mut = Read_data()
	temp = [0.0001, 1, 10, 100, 1000]
	num_samps = 50000
	alpha = 7
	gamma = 0.1
	kappa = 0.05 # Same to the source code

	# test_lexicon = torch.zeros(0,2)
	# # test_lexicon_mat = Get_intention_object()
	# print(Evaluate_lexicon(test_lexicon, corpus_objects, corpus_words, alpha, len(world_words), len(world_objects), lexicons_mat[i], intention_mats))

	intention_mats = []
	for i in range(len(corpus_objects)):
		curr_situ_objects = corpus_objects[i].tolist()
		intention_mats += [Get_intention_object(len(curr_situ_objects), gamma)]

	# Initialize lexicons, evaluating initial lexicons
	lexicons = Initialize_lexicon(len(temp), world_words, world_objects)
	# lex_test = torch.zeros(1,2).long()
	# lex_test_mat = Get_object_words(lex_test, corpus_objects, corpus_words, kappa, len(world_words), len(world_objects))
	# prob_test = Evaluate_lexicon(lex_test, corpus_objects, corpus_words, alpha, len(world_words), len(world_objects), lex_test_mat, intention_mats)
	# print(prob_test)

	lexicons_mat = []
	for i in range(len(lexicons)):
		lexicons_mat += [Get_object_words(lexicons[i], corpus_objects, corpus_words, kappa, len(world_words), len(world_objects))]
	probs = torch.zeros(len(lexicons))
	for i in range(len(lexicons)):
		probs[i] = Evaluate_lexicon(lexicons[i], corpus_objects, corpus_words, alpha, len(world_words), len(world_objects), lexicons_mat[i], intention_mats)
	print(probs) # Initilal probabilities


	for i in range(num_samps):
		if i % 50 == 0:
			print("Current episode: %d"%(i))
		for T in range(len(temp)):
			new_lexicon, word_idx, obj_idx, mut_type = Mutate(lexicons[T], world_mut)
			new_lexicon_mat = Get_object_words(new_lexicon, corpus_objects, corpus_words, kappa, len(world_words), len(world_objects))
			new_prob = Evaluate_lexicon(new_lexicon, corpus_objects, corpus_words, alpha, len(world_words), len(world_objects), new_lexicon_mat, intention_mats)
			if torch.rand(1) < ((new_prob - probs[T]) / temp[T]):
				lexicons[T] = new_lexicon
				lexicons_mat[T] = new_lexicon_mat
				probs[T] = new_prob
				print("Lexicons on temperature %f was changed." %(temp[T]))
				print("pair %s   %s, method: %s" %(world_words[word_idx], world_objects[obj_idx], mut_type))
		if i % 5 == 0: # Breed
			lex_a_idx = torch.randint(0, len(temp), [1])
			lex_b_idx = torch.randint(0, len(temp), [1])
			if lex_a_idx != lex_b_idx:
				new_lexicon = Breed(lexicons[lex_a_idx], lexicons[lex_b_idx], temp[lex_a_idx], temp[lex_b_idx])
				new_lexicon_mat = Get_object_words(new_lexicon, corpus_objects, corpus_words, kappa, len(world_words), len(world_objects))
				new_prob = Evaluate_lexicon(new_lexicon, corpus_objects, corpus_words, alpha, len(world_words), len(world_objects), new_lexicon_mat, intention_mats)
				if torch.rand(1) < ((new_prob - probs[lex_a_idx]) / temp[lex_a_idx]):
					lexicons[lex_a_idx] = new_lexicon
					lexicons_mat[lex_a_idx] = new_lexicon_mat
					probs[lex_a_idx] = new_prob
					print("Lexicons on temperature %f was changed due to breed." %(temp[lex_a_idx]))
		if i % 100 == 0: # Show current result
			print("------ Episode %d ------" %(i))
			print("current score:")
			for j in range(len(temp)):
				print("Temperature %f score: %f" %(temp[j], probs[j]))
		if i % 100 == 0:
			print("------ current lexicon: ------")
			for j in range(len(temp)):
				print("--Temperature %f lexicon:--" %(temp[j]))
				print(lexicons[j])
				for k in range(lexicons[j].shape[0]):
					print("%s   %s" %(world_words[lexicons[j][k][0]], world_objects[lexicons[j][k][1]]))

			





