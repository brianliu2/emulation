# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:17:44 2017

@author: xliu
"""
import numpy as np
import scipy as sp
from numpy.matlib import repmat
from update_gpparam import getCorrMat
from preprocess_initialization import *
from sampled_mean_generator import *

def calibrator(train_inputs, train_inputs_norm, train_outputs, train_outputs_log, emulator_param, V_mat, beta_mat, alpha, const, nugget, field_data, start_field_t, end_field_t, emulation_burnin, iter_factor, sd_prop_vec, prop_factor, cali_likeli_sampling_iter, AR_order, likelihood, sampleZ_iter_num):
	
	'''
	we need to perform a sanity check to see if the length of field data equals to the time frame
	setup by start_field_t: end_field_t,
	'''	
	if len(field_data) != len(range(start_field_t-1, end_field_t)):
		raise ValueError('Length of field data should be identical to time frame setup by start_field_t and end_field_t.')
	
	# for sampling mean of likelihood
	sample_Z_input_args = {}
	sample_Z_input_args['field_tspan'] = range(start_field_t-1, end_field_t)
	sample_Z_input_args['AR_order'] = AR_order
	sample_Z_input_args['sampleZ_iter_num'] = sampleZ_iter_num
	# read-in the built emulator
	emulator_after_burnin = emulator_param[:, :, emulation_burnin:]
	V_after_burnin = V_mat[:, emulation_burnin:]
	beta_after_burnin = beta_mat[:, emulation_burnin:]
	
	emulation_iter  = V_after_burnin.shape[1]
	calibraion_iter = iter_factor * emulation_iter
		
	sample_Z_input_args['train_inputs'] = train_inputs
	sample_Z_input_args['train_outputs_log'] = train_outputs_log
	len_train_outputs = train_outputs_log.shape[0]
	
	train_inputs_aug = input_augment(train_inputs_norm)
	sample_Z_input_args['train_inputs_aug'] = train_inputs_aug
	
	# pre-allocate the parameter space
	dim_param = emulator_param.shape[0]-2
	param_posterior = np.zeros((calibraion_iter, dim_param))
	
	param_temp = 0.5 * np.ones((1, dim_param))
	#param_temp_aug = input_augment(param_temp)
	
	param_posterior[0, :] = param_temp
	
#	sample_Z_input_args['param_t'] = param_t
	
	# proposal covariance for MH step
	sd_prop = prop_factor * sd_prop_vec
	
	# acceptance counter
	accept_eta = 0
	
	# correlation matrix using the last beta_cur
	beta_T = beta_after_burnin[:, -1]
	corrMat_train_inputs = getCorrMat(train_inputs_norm, beta_T, alpha)
	corrMat_train_inputs_inv = np.linalg.inv(corrMat_train_inputs)
	sample_Z_input_args['corrMat_train_inputs_inv'] = corrMat_train_inputs_inv
	
	# initialize index for DeModularization
	deModu_idx = np.random.choice(emulation_iter, size = 1)[0]
	
	# select Phi, Beta and V according to index selected
	beta_t = beta_after_burnin[:, deModu_idx].T
	emulator_param_t = emulator_param[:, :, deModu_idx]
	V_t = V_mat[:, deModu_idx]
	
	sample_Z_input_args['emulator_param_t'] = emulator_param_t
	sample_Z_input_args['V_t'] = V_t
	
	# initialize
	param_temp_extend = repmat(param_temp, m = train_inputs_norm.shape[0], n=1)
	
	# to calculate pair-wise distance between eta and x, we need to replicate mat of beta, then concatenate
	beta_t_extend = repmat(beta_t, m = train_inputs_norm.shape[0], n = 1)
	para_beta_t_extend = np.c_[param_temp_extend, beta_t_extend]	
	
	# calculate the correlation between the current samples and train inputs	
	rho_lst_t = list(map(lambda para, x: (para[6:]*np.sqrt(((para[:6]-x)**2))).sum(), para_beta_t_extend, train_inputs_norm))		
	rho_t = np.array(rho_lst_t)
	sample_Z_input_args['rho'] = rho_t
	
	# pre-allocate memory for storing sampled means
	sampled_mu_likelihood = np.zeros((train_outputs.shape[0], calibraion_iter))
	
	#---------------------------------------------------------------------------------------#
	#                            	                                                          #
	#                            	                                                          #
	#                            	                                                          #
	#	                                Main Loop                                           #
	#                            	                                                          #
	#                            	                                                          #
	#                            	                                                          #
	#---------------------------------------------------------------------------------------#
	'''
	Below are the main body for calibration, in the version mplemented here, we implemented
	the demodularization to selective draw emulator parameters. In addition, the field data 
	could be modelled as Poisson or negative binomial. It will be much straightforward that
	if the field data is Poisson, because its has the same mean and variance lambda.
	More complicate case is negative binomial, we use gamma to calculate the factorial of 
	negative binomial, and paramerize the distribution in a different way. Details can be found
	in our paper.
	'''	
	accept_cnt = 0
	for ical in range(calibraion_iter):
		param_temp = param_posterior[ical, :] 
		
		# randomly generated index
		deModu_next_idx = np.random.choice(emulation_iter, size = 1)[0]
		
		# set temp beta and phi according to index
		beta_t_temp = beta_after_burnin[:, deModu_next_idx].T
		emulator_param_t_temp = emulator_after_burnin[:, :, deModu_next_idx]
		V_t_temp = V_after_burnin[:, deModu_next_idx]
		
		# update the pair distance between parameter and training inputs according to updated index
		beta_t_temp_extend = repmat(beta_t_temp, m = train_inputs_norm.shape[0], n = 1)
		para_t_beta_t_temp_extend = np.c_[param_t_extend, beta_t_temp_extend]	
		rho_lst_t_temp = list(map(lambda para, x: (para[6:]*np.sqrt(((para[:6]-x)**2))).sum(), para_t_beta_t_temp_extend, train_inputs_norm))		
		rho_t_temp = np.array(rho_lst_t_temp)
		
		# augment the current parameter samples
		param_temp_aug = input_augment(param_temp)
		sample_Z_input_args['param_temp_aug'] = param_temp_aug
		
		# Depend on the type of likelihood, we use the different sampling step
		
		mean_generator_results = sampledZGenerator(field_data, likelihood, ical, sample_Z_input_args, nugget, const)		
		input_args_dict['sampled_Z_log'] = mean_generator_results['predMu_log']
		input_args_dict['pdfMuLogGivenM_S'] = mean_generator_results['pdfMuLogGivenMtSt']
		input_args_dict['nb_dispersion'] = mean_generator_results['nb_dispersion']
		
		
		
		pdfMuLogGivenM_S_star = pdfMuLogGivenM_S_evaluate(param_aug, emulator_param_t_temp, V_t_temp, train_outputs_log, train_inputs_aug, rho_t_temp, sample_Z_input_args['field_tspan'], corrMat_train_inputs_inv, AR_order, input_args_dict['sampled_Z_log'], weekly=True)
		
		accpRate = np.log(pdfMuLogGivenM_S_star[1:, :].sum()) - np.log(input_args_dict['pdfMuLogGivenM_S'][1:, :],sum())
		
		if np.log(np.random.uniform(size = 1)) < accpRate:
			emulator_param_t = emulator_param_t_temp
			V_t = V_t_temp
			beta_t = beta_t_temp
			rho_t = rho_t_temp
			sample_Z_input_args['V_t'] = V_t
			sample_Z_input_args['rho'] = rho_t
			sample_Z_input_args['emulator_param_t'] = emulator_param_t
			# update sampled mean and dispersion according to updated emulator parameters
			mean_generator_results = sampledZGenerator(field_data, likelihood, ical, sample_Z_input_args, nugget, const)
			input_args_dict['sampled_Z_log'] = mean_generator_results['predMu_log']
			input_args_dict['pdfMuLogGivenM_S'] = mean_generator_results['pdfMuLogGivenMtSt']
			input_args_dict['nb_dispersion'] = mean_generator_results['nb_dispersion']
		# save the sampled mean
		sampled_mu_likelihood[:, ical] = mean_generator_results['predMu']
		
		# **************** Propose value for parameters, and accept/reject according to likelihood **************** #
		bigM = MT_fcn(input_args_dict['sampled_Z_log'], train_outputs_log, emulator_param_t, corrMat_train_inputs_inv, rho_t, train_inputs_aug, param_temp_aug, AR_order)
		bigS = ST_fcn(input_args_dict['sampled_Z_log'], V_t, corrMat_train_inputs_inv, rho_t, AR_order)
		
		# propose value for parameters
		logit_param_temp_star = np.log(param_temp/(1-param_temp)) + (sd_prop * np.random.randn(6,1)).T
		param_temp_star = 1 / (1 + np.exp(-logit_param_temp_star))
		param_temp_aug_star = input_augment(param_temp_star)
		
		param_temp_star_extend = repmat(param_temp_star, m = train_inputs_norm.shape[0], n=1)
	
		# to calculate pair-wise distance between eta and x, we need to replicate mat of beta, then concatenate
		beta_t_extend = repmat(beta_t, m = train_inputs_norm.shape[0], n = 1)
		para_beta_temp_star_extend = np.c_[param_temp_star_extend, beta_t_extend]	
		
		# calculate the correlation between the current samples and train inputs	
		rho_lst_temp_star = list(map(lambda para, x: (para[6:]*np.sqrt(((para[:6]-x)**2))).sum(), para_beta_temp_star_extend, train_inputs_norm))		
		rho_temp_star = np.array(rho_lst_temp_star)
		
		bigM_star = MT_fcn(input_args_dict['sampled_Z_log'], train_outputs_log, emulator_param_t, corrMat_train_inputs_inv, rho_temp_star, train_inputs_aug, param_temp_aug_star, AR_order)
		bigS_star = ST_fcn(input_args_dict['sampled_Z_log'], V_t, corrMat_train_inputs_inv, rho_temp_star, AR_order)
		
		sampled_Z_log = input_args_dict['sampled_Z_log']
		
		# calculate the log-acceptance ratio
		accept_param = -0.5*(((sampled_Z_log[1:, :] - bigM_star)**2/bigS_star).sum()-((sampled_Z_log[1:, :] - bigM)**2/bigS).sum())\
		-0.5*(np.log(bigS_star).sum())+0.5*(np.log(bigS).sum())\
		+np.log(param_temp_star).sum()-np.log(param_temp).sum()\
		+np.log(1-param_temp_star).sum()-np.log(1-param_temp).sum()
		
		if np.log(np.random.uniform(size = 1)) < accept_param:
			param_posterior[ical+1, :] = param_temp_star
			sample_Z_input_args['rho'] = rho_temp_star
			accept_cnt += 1
		else:
			param_posterior[ical+1, :] = param_posterior[ical, :]
	return param_posterior, sampled_mu_likelihood, accept_cnt
	
	



if __name__ == '__main__':
	print('ok')
	




































