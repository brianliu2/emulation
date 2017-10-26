# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 19:30:50 2017

@author: xliu
"""
from numpy.matlib import repmat
import numpy as np

def tvar_multivariate_G(train_outputs_log, \
		AR_order, delta_w, delta_v, mu_0, Cov_0, \
		s_0, n_0, corrMat, state_covMat_indicator, train_inputs_aug):
	
	T_train_outputs, num_train_outputs = train_outputs_log.shape
	
	# MCMC outputs
	mu  = repmat(mu_0, m = 1, n = T_train_outputs)
	s_mat   = s_0 * np.ones((1, T_train_outputs))
	n_mat   = n_0 * np.ones((1, T_train_outputs))
	Cov = np.tile(Cov_0[:, :, np.newaxis], [1, 1, T_train_outputs])
	
	# initialize
	mu_t = mu_0
	Cov_t = Cov_0
	s_t = s_0
	n_t = n_0
	
	# forward filtering
	for t in range(AR_order, T_train_outputs):
		Cov_t = np.dot(np.dot(state_covMat_indicator, Cov_t), state_covMat_indicator.T)
		mu_t  = np.dot(state_covMat_indicator, mu_t)
		
		if AR_order > 1:
			ind = range(t-1, t-AR_order, -1)
			Fprime = np.hstack((train_outputs_log[ind, :].T.reshape(-1, 1), train_inputs_aug))
		else:
			Fprime = np.hstack((train_outputs_log[t-1, :].T.reshape(-1, 1), train_inputs_aug))
		
		A = np.divide(np.dot(Cov_t, Fprime.T), delta_w)
		
		Q = np.dot(Fprime, A) + s_t * corrMat
		
		A = np.dot(np.linalg.inv(Q), A.T).T
		
		e = train_outputs_log[t, :].T.reshape(-1,1) - np.dot(Fprime, mu_t)

		mu_t = mu_t + np.dot(A, e)
		
		mu[:, t] = mu_t.ravel()
		
		ddt = n_t * s_t
		
		ddt = delta_v * ddt + s_t * np.dot(e.T, np.dot(np.linalg.inv(Q), e))
		
		n_t = delta_v * n_t + num_train_outputs
		
		s_t = ddt / n_t
		
		n_mat[:, t] = n_t.ravel()
		s_mat[:, t] = s_t.ravel()
		
		Cov_t = (s_t/s_mat[:, t-1]) * (np.divide(Cov_t, delta_w) - np.dot(np.dot(A, Q), A.T))
		
		Cov_t = (Cov_t + Cov_t.T) / 2
		
		Cov[:, :, t] = Cov_t
	
	# loop-over the time steps
	mu[:, :AR_order] = repmat(mu[:, AR_order], 1, AR_order).T
	Cov[:, :, :AR_order] = np.tile(Cov[:, :, AR_order], [1, 1, AR_order]).reshape(Cov.shape[0], Cov.shape[1], 1)
	n_mat[:, :AR_order] = repmat(n_mat[:, AR_order], 1, AR_order)
	s_mat[:, :AR_order] = repmat(s_mat[:, AR_order], 1, AR_order)	
	
	return mu, Cov, s_mat, n_mat


def tvar_multivariate2_G(train_outputs_log, AR_order, discount_factor, mu_0, Cov_0, \
	var_t, corrMat, state_covMat_indicator, state_covMat_indicator_inv, train_inputs_aug):
	
	delta_w = discount_factor[0]
	
	T_train_outputs, num_train_outputs = train_outputs_log.shape
	
	Phi = np.zeros((mu_0.shape[0], T_train_outputs))
	Err = np.zeros((T_train_outputs, num_train_outputs))
	
	mu  = repmat(mu_0, m = 1, n = T_train_outputs)
	Cov = np.tile(Cov_0[:, :, np.newaxis], [1, 1, T_train_outputs])
	
	mu_t = mu_0
	Cov_t = Cov_0
	
	for t in range(AR_order, T_train_outputs):
		Cov_t = np.dot(np.dot(state_covMat_indicator, Cov_t), state_covMat_indicator.T)
		mu_t  = np.dot(state_covMat_indicator, mu_t)
		
		if AR_order > 1:
			ind = range(t-1, t-AR_order, -1)
			Fprime = np.hstack((train_outputs_log[ind, :].T.reshape(-1, 1), train_inputs_aug))
		else:
			Fprime = np.hstack((train_outputs_log[t-1, :].T.reshape(-1, 1), train_inputs_aug))
		
		A = np.divide(np.dot(Cov_t, Fprime.T), delta_w)
		
		Q = np.dot(Fprime, A) + var_t[:, t] * corrMat
		
		A = np.dot(np.linalg.inv(Q), A.T).T
		
		e = train_outputs_log[t, :].T.reshape(-1,1) - np.dot(Fprime, mu_t)

		mu_t = mu_t + np.dot(A, e)
		
		mu[:, t] = mu_t.ravel()
		
		Cov_t = (np.divide(Cov_t, delta_w) - np.dot(np.dot(A, Q), A.T))
		
		Cov_t = (Cov_t + Cov_t.T) / 2
		
		Cov[:, :, t] = Cov_t
	
	# End-of-loop
	mu[:, :AR_order] = repmat(mu[:, AR_order], 1, AR_order).T
	Cov[:, :, :AR_order] = np.tile(Cov[:, :, AR_order], [1, 1, AR_order]).reshape(Cov.shape[0], Cov.shape[1], 1)
	
	# backward sampling
	Phi[:, -1] = np.random.multivariate_normal(mean = mu[:, -1], cov = Cov[:, :, -1], size = (mu_0.shape[1]))
	
	for t in range(T_train_outputs-2, AR_order-1, -1):
		mu_t = (1 - delta_w) * mu[:, t] + delta_w * np.dot(state_covMat_indicator_inv, Phi[:, t+1])
		
		if t > AR_order:
			if AR_order > 1:
				ind = range(t-1, t-AR_order, -1)
				Fprime = np.hstack((train_outputs_log[ind, :].T.reshape(-1, 1), train_inputs_aug))
			else:
				Fprime = np.hstack((train_outputs_log[t-1, :].T.reshape(-1, 1), train_inputs_aug))
			Err[t+1, :] = train_outputs_log[t+1, :] - np.dot(Phi[:, t+1].T , Fprime.T)
		
		Cov_t = (1 - delta_w) * Cov[:, :, t]
		Cov_t = (Cov_t + Cov_t.T) / 2
		Phi[:, t] = np.random.multivariate_normal(mean = mu_t, cov = Cov_t, size = (mu_0.shape[1]))
	
	Err[AR_order, :] = train_outputs_log[AR_order, :] - np.dot(Phi[:, AR_order].T, Fprime.T)
	
	return Phi, Err
	
if __name__ == '__main__':
	print('ok')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
