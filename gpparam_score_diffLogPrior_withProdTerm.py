# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:53:10 2017

@author: xliu
"""
import numpy as np

def gpparam_score_diffLogPrior_withProdTerm(corrMat, beta_t, train_inputs_norm, Err, var, AR_order):

	T_Err, num_Err = Err.shape
	
	logPrior = 0
	
	dim = beta_t.shape[0]
	
	for d in range(dim):
		logPrior += -beta_t[d, :] + 0.9 * np.log(1 - np.exp(-beta_t[d,:]/4))
	
	corrMat_det = np.linalg.det(corrMat)
	
	corrMat_inv = np.linalg.inv(corrMat)
	corrMat_inv = (corrMat_inv + corrMat_inv.T)/2
	
	logLike = -0.5 * (T_Err - AR_order) * np.log(corrMat_det)
	
	for t in range(AR_order, T_Err):
		var_t = var[:, t]
		e = Err[t, :] / np.sqrt(var_t)
		logLike += -0.5 * np.dot(np.dot(e, corrMat_inv), e.T) - num_Err/2 * np.log(var_t)
	
	logJacobi = np.sum(np.log(beta_t))
	
	score = logPrior + logLike + logJacobi
												
	return score


if __name__ == '__main__':
	print('ok')