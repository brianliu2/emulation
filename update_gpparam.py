# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:22:39 2017

@author: xliu
"""
import numpy as np
from gpparam_score_diffLogPrior_withProdTerm import *

def update_gpparam(alpha, beta, train_inputs_norm, Err, var, AR_order, score):
	
	beta_row, beta_col = beta.shape
	
	log_beta_star = np.log(beta) + 0.01 * np.random.randn(beta_row, beta_col)
	beta_star = np.exp(log_beta_star)
	
	corrMat = getCorrMat(train_inputs_norm, beta_star, alpha)
	score_star = gpparam_score_diffLogPrior_withProdTerm(corrMat, beta_star, train_inputs_norm, Err, var, AR_order)
	
	acceptance = score_star - score

	if np.log(np.random.uniform(size = 1)) < acceptance:
		beta_update = beta_star
	else:
		beta_update = beta
	
	return beta_update
	
def getCorrMat(inputs, beta, alpha):
	num_inputs, dim_inputs = inputs.shape
	corrMat = np.zeros((num_inputs, num_inputs))
	for i in range(num_inputs):
		for j in range(num_inputs):
			element = 0
			for d in range(dim_inputs):
				element = element + beta[d] * np.abs(inputs[i, d] - inputs[j, d]) ** alpha[d]
			corrMat[i, j] = np.exp(-element)
			corrMat[j, i] = corrMat[i, j]
	return corrMat	
	


if __name__ == '__main__':
	print('ok')

























