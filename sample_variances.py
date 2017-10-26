# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:00:14 2017

@author: xliu
"""
import numpy as np

def sample_variances(n_vect, s_vect, AR_order, discount_factor):
	delta_w = discount_factor[1]
	
	time_length = n_vect.shape[1]
	var_vec = np.zeros((1, time_length))
	
	n_t = n_vect[:, -1]
	
	d_t = n_vect[:, -1] * s_vect[:, -1]
	
	var_T = np.random.gamma(shape = n_t/2, scale = 2/d_t, size = 1)
	
	var_vec[:, -1] = var_T
	
	for t in range(time_length-2, AR_order-1, -1):
		n_t = (1 - delta_w) * n_vect[:, t]
		d_t = n_vect[:, t] * s_vect[:, t]
		var_T = delta_w * var_T + np.random.gamma(shape = n_t/2, scale = 2/d_t, size = 1)
		var_vec[:, t] = 1 / var_T
	
	var_vec[:, :AR_order] = var_vec[:, AR_order]
	
	return var_vec


if __name__ == '__main__':
	print('ok')

















































