# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:17:44 2017

@author: xliu
"""
import numpy as np
import scipy as sp
from numpy.matlib import repmat
from update_gpparam import getCorrMat

def calibrator(train_inputs_norm, emulator_param, V_mat, beta_mat, alpha, const, nugget, field_data, emulation_burnin, iter_factor, sd_prop_vec, prop_factor, cali_likeli_sampling_iter):
	emulator_after_burnin = emulator_param[:, :, emulation_burnin:]
	V_after_burnin = V_mat[:, emulation_burnin:]
	beta_after_burnin = beta_mat[:, emulation_burnin:]
	
	calibraion_iter = iter_factor * V_after_burnin.shape[1]
	
	# pre-allocate the parameter space
	dim_param = emulator_param.shape[0]-2
	param_posterior = np.zeros((calibraion_iter, dim_param))
	param_posterior[0, :] = 0.5 * np.ones((1, dim_param))
	
	# proposal covariance for MH step
	sd_prop = prop_factor * sd_prop_vec
	
	# acceptance counter
	accept_eta = 0
	
	# correlation matrix using the last beta_cur
	beta_T = beta_after_burnin[:, -1]
	corrMat_train_inputs = getCorrMat(train_inputs_norm, beta_T, alpha)
	corrMat_train_inputs_inv = np.linalg.inv(corrMat_train_inputs)
	
	# initialize index for DeModularization
	deModu_idx = np.random.choice(calibraion_iter, size = 1)
	
	# select Phi, Beta and V according to index selected
	beta_i = beta_after_burnin[:, deModu_idx]
	emulator_param_i = emulator_param[:, :, deModu_idx]
	V_i = V_mat[:, deModu_idx]
	
	# calculate the correlation between the current samples and train inputs
	eta_extend = repmat(eta, m = 1, n=train_inputs_norm.shape[0]).T
	rho_lst = list(map(lambda eta, x, beta: (beta * np.sqrt(((eta-x)**2))).sum(), eta_extend, train_inputs_norm, beta_i))
	rho = np.array(rho_lst)
			

	
	
	return param_posterior
	
	



if __name__ == '__main__':
	print('ok')
	




































