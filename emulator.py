# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:23:45 2017

@author: xliu
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
import pickle
from sklearn.externals import joblib
import os
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from update_gpparam import *
from tvar_multivariate_G import *
from sample_variances import *
from preprocess_initialization import *
from gpparam_score_diffLogPrior_withProdTerm import *
from calibrator import *

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20

class emulator_cls(object):
	def __init__(self, data, nugget = 0.5, const = 5, AR_order = 1, discount_factor = 0.9, \
				  emulation_iter = 1200, emulation_burnin = 200, MH_within_Gibbs_iter = 25, \
				  beta_init = 1, alpha_val = 1.999, cali_iter_factor = 100, cali_likeli_sampling_iter = 5):
		self.train_inputs  = data['train_inputs']
		self.train_outputs = data['train_outputs']
		self.valid_inputs  = data['valid_inputs']
		self.valid_outpus  = data['valid_outputs']
		
		self.num_train_inputs = self.train_inputs.shape[0]
		self.dim_train_inputs = self.train_inputs.shape[1]
		
		self.train_time_length = self.train_outputs.shape[0] # train_outputs: T * N
		self.train_tspan = range(self.train_time_length)
		
		self.nugget = 0.5
		self.const  = 5
		self.AR_order = 1
		
		self.train_outputs_log = log_trans(self.train_outputs, nugget = self.nugget, const = self.const)
		self.valid_outputs_log = log_trans(self.valid_outpus, nugget = self.nugget, const = self.const)
		
		self.train_inputs_norm = input_normalize_denormalize(self.train_inputs, is_normalize=True)
		self.valid_inputs_norm = input_normalize_denormalize(self.valid_inputs, is_normalize=True)
		
		self.train_inputs_aug  = input_augment(self.train_inputs_norm)
		
		self._dim_train_inputs_aug = self.train_inputs_aug.shape[1]
		self._state_covMat_indicator = np.eye(self.AR_order + self._dim_train_inputs_aug)
		self._state_covMat_indicator_inv = np.linalg.inv(self._state_covMat_indicator)
		
		# prior settings
		self._mu_0 = np.vstack((np.ones((self.AR_order, 1)) ,np.ones((self._dim_train_inputs_aug, 1))))
		self._Cov_0 = 5 * np.eye(self.AR_order + self._dim_train_inputs_aug)
		self._n_0 = 1
		self._s_0 = 1
		self.discount_factor = np.array([discount_factor, discount_factor])
		
		self.emulation_iter = emulation_iter
		self.emulation_burnin = emulation_burnin
		self.MH_within_Gibbs_iter = MH_within_Gibbs_iter
		
		self.emulator_param = np.zeros((len(self._mu_0), len(self.train_tspan), self.emulation_iter))
		self.V_mat = np.zeros((len(self.train_tspan), self.emulation_iter))
		self.Err_mat = np.zeros((len(self.train_tspan), self.num_train_inputs, self.emulation_iter))
		self.beta_mat = np.zeros((self.dim_train_inputs, self.emulation_iter))
		
		self.beta_cur = beta_init * np.ones((self.dim_train_inputs, 1))
		self.beta_mat[:, 0] = self.beta_cur.ravel()
		
		#self.alpha = alpha_val * np.ones_like(self.beta_cur)
		self.alpha = alpha_val
		
		# setting for calibration		
		self.cali_iter_factor = cali_iter_factor
		self.sd_prop_vec = np.array([10.1365629, 0.6130522, 17,4841793, 0.5461787, 0.5820872, 0.553091])
		self.cali_likeli_sampling_iter = cali_likeli_sampling_iter
		#self.likelihood = likelihood
#	def getCorrMat(self, inputs):
#		num_inputs, dim_inputs = inputs.shape
#		corrMat = np.zeros((num_inputs, num_inputs))
#		for i in range(num_inputs):
#			for j in range(num_inputs):
#				element = 0
#				for d in range(dim_inputs):
#					element = element + self.beta_cur[d] * np.abs(inputs[i, d] - inputs[j, d]) ** self.alpha[d]
#				corrMat[i, j] = np.exp(-element)
#				corrMat[j, i] = corrMat[i, j]
#		return corrMat
	def getCorrMat(self, inputs):
		pairwise_dists = squareform(pdist(inputs, 'seuclidean', V = 1/self.beta_cur.ravel()))
		corrMat = sp.exp(-pairwise_dists ** self.alpha)
		return corrMat
		
	def _tvar_multivariate_G(self, corrMat):
		
		mu_update, Cov_update, n_update, s_update = tvar_multivariate_G(self.train_outputs_log, \
		self.AR_order, self.discount_factor[0], self.discount_factor[1], self._mu_0, self._Cov_0, \
		self._s_0, self._n_0, corrMat, self._state_covMat_indicator, self.train_inputs_aug)
		
		return mu_update, Cov_update, n_update, s_update
	
	def _sample_variance(self, n_t, s_t):
		var_vec = sample_variances(n_t, s_t, self.AR_order, self.discount_factor)
		return var_vec
	
	def _tvar_multivariate2_G(self, corrMat, var):
		
		Phi_t, Err_t = tvar_multivariate2_G(self.train_outputs_log, \
		self.AR_order, self.discount_factor, self._mu_0, self._Cov_0, \
		var, corrMat, self._state_covMat_indicator, self._state_covMat_indicator_inv, self.train_inputs_aug)
		
		return Phi_t, Err_t
	
	def _gpparam_score_diffLogPrior_withProdTerm(self, Err, var):
		beta_t = self.beta_cur
		train_inputs_norm = self.train_inputs_norm
		AR_order = self.AR_order
		
		corrMat = self.getCorrMat(train_inputs_norm)
		
		score = gpparam_score_diffLogPrior_withProdTerm(corrMat, beta_t, train_inputs_norm, Err, var, AR_order)
		return score
		
	def fit(self, aggregated = False, total_weeks = None):
		# if we want to use the aggregated weekly data, then further process the training/validation dataset
		if aggregated:
			if total_weeks is None:
				raise ValueError('total weeks need to be specified.')
			self.train_outputs, self.valid_outpus = aggregated_fcn(self.train_outputs, self.valid_outpus, total_weeks)
			self.train_time_length = self.train_outputs.shape[0] # train_outputs: T * N
			self.train_tspan = range(self.train_time_length)
			self.train_outputs_log = log_trans(self.train_outputs, nugget = self.nugget, const = self.const)
			self.valid_outputs_log = log_trans(self.valid_outpus, nugget = self.nugget, const = self.const)
			self.emulator_param = np.zeros((len(self._mu_0), len(self.train_tspan), self.emulation_iter))
			self.V_mat = np.zeros((len(self.train_tspan), self.emulation_iter))
			self.Err_mat = np.zeros((len(self.train_tspan), self.num_train_inputs, self.emulation_iter))
			
		# main loop to fit emulator
		for ifit in range(self.emulation_iter):
			print('Iter: {0}\tBeta:{1}'.format(ifit, self.beta_cur.ravel()))
			#print(self.beta_cur)
			corrMat = self.getCorrMat(self.train_inputs_norm)
			mu_t, Cov_t, s_t, n_t = self._tvar_multivariate_G(corrMat)
			var_t = self._sample_variance(n_t, s_t)
			emulator_param_t, Err_t = self._tvar_multivariate2_G(corrMat, var_t)
			score_t = self._gpparam_score_diffLogPrior_withProdTerm(Err_t, var_t)
			
			for Gibbs_MH_iter in range(self.MH_within_Gibbs_iter):
				self.beta_cur = update_gpparam(self.alpha, self.beta_cur, self.train_inputs_norm, Err_t, var_t, self.AR_order, score_t)
			
			self.V_mat[:, ifit] = var_t
			self.emulator_param[:, :, ifit] = emulator_param_t
			self.Err_mat[:, :, ifit] = Err_t
			self.beta_mat[:, ifit] = self.beta_cur.ravel()
	
	def calibrate(self, field_data, start_field_t, end_field_t, likelihood, cali_iter_factor=100, prop_factor = 0.11):
		param_posterior = calibrator(self.train_inputs, self.train_inputs_norm, self.train_outputs, self.train_outputs_log, self.emulator_param, self.V_mat, self.beta_mat, self.alpha, self.const, self.nugget, field_data, start_field_t, end_field_t, self.emulation_burnin, cali_iter_factor, self.sd_prop_vec, prop_factor, self.cali_likeli_sampling_iter, self.AR_order, likelihood, sampleZ_iter_num = 5)
		return param_posterior
		
if __name__ == '__main__':
	data_path = os.path.join('/Users/xliu/Documents/MRC/Work/Program/','emulator/python_version/emulation_python_ver/')
	train_inputs  = data_path + 'LHCDesign_training.txt'
	#train_output = data_path + 'Outputs_training.txt'
	train_outputs = data_path + 'training_outputs_andre.txt'
	valid_inputs  = data_path + 'LHCDesign_validation.txt'
	#valid_output = data_path + 'Outputs_validation.txt'
	valid_outputs = data_path + 'validation_outputs_andre.txt'
	field_data    = data_path + 'andre_agg_estimates_London_python.txt'	
	
	train_inputs  = read_data_from_txt(train_inputs, is_output = False)
	train_outputs = read_data_from_txt(train_outputs, is_output = True)
	valid_inputs  = read_data_from_txt(valid_inputs, is_output = False)
	valid_outputs = read_data_from_txt(valid_outputs, is_output = True)
	field_data    = read_data_from_txt(field_data, is_output=False)	
	
	data = {'train_inputs': train_inputs, 'train_outputs': train_outputs,
	        'valid_inputs': valid_inputs, 'valid_outputs': valid_outputs}
#									
	emulator = emulator_cls(data)
	emulator.fit(aggregated=True, total_weeks = 47)
#	
	phi_4_025Perc = np.percentile(emulator.emulator_param[3, :, emulator.emulation_burnin:], 2.5, axis=1)
	phi_4_50Perc = np.percentile(emulator.emulator_param[3, :, emulator.emulation_burnin:], 50, axis=1)
	phi_4_975Perc = np.percentile(emulator.emulator_param[3, :, emulator.emulation_burnin:], 97.5, axis=1)	
#	
	plt.plot(phi_4_025Perc);
	plt.plot(phi_4_50Perc);
	plt.plot(phi_4_975Perc);	

	model_name = 'emulator_andre_data.sav'
	joblib.dump(emulator, data_path+model_name)
#	
#	# load model
	emulator_built = joblib.load(model_name)
	print(emulator_built.emulator_param.shape)
#	param_posterior = emulator_built.calibrate(field_data, start_field_t = 5, end_field_t = 47, likelihood = 'negative binomial')
	print('ok')












