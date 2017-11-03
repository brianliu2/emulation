# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 16:06:49 2017

@author: xliu
"""

import numpy as np
from scipy.stats import norm, gamma
'''
Poisson:
	Input -->
	ical = 0:
		etaTemp, X, hEta, Phii, Vi, aggregated_train_log, hX, nugget, const, rho,SigmaInv,week_span,Ptvar, aggregated_z, Bi, sigmaLogMu, ical, 0, 0
	ical > 0:
		etaTemp, X, hEta, Phii, Vi, aggregated_train_log, hX, nugget, const, rho,SigmaInv,week_span,Ptvar, aggregated_z, Bi, sigmaLogMu, ical, sampledZLog, pdfMuLogGivenM_S_og
	
	Output -->
		sampledZ, sampledZLog, acceptRatio, pdfMuLogGivenM_S_og, nb_dispersion, acceptCnt_dispersion
NB: 
	Input -->
	ical = 0:	
		etaTemp, X, hEta, Phii, Vi, aggregated_train_log, hX, nugget, const, rho,SigmaInv,week_span,Ptvar, aggregated_z, Bi, ical, 0, 0, nb_dispersion_previous
	ical > 0:
		etaTemp, X, hEta, Phii, Vi, aggregated_train_log, hX, nugget, const, rho,SigmaInv,week_span,Ptvar, aggregated_z, Bi, ical, sampledZLog, pdfMuLogGivenM_S_og, nb_dispersion_previous
	
	Output -->
		sampledZ, sampledZLog, acceptRatio, pdfMuLogGivenM_S_og
'''

def sampledZGenerator(field_data, likelihood_dist, cali_iter, input_args_dict, nugget, const):
	# do calculation according to different types of likelihood
	if likelihood_dist == "Poisson":
		Pois_mean_generator_result = Pois_sampledZGenerator()
		return Pois_mean_generator_result
	elif likelihood_dist == 'negative binomial':
		NB_mean_generator_result = NB_sampledZGenerator(field_data, cali_iter, input_args_dict, nugget, const)
		return NB_mean_generator_result
	elif likelihood_dist == 'normal':	
		raise ValueError('We do not need to sample mean value in normal likelihood, since the conjugacy is not lost.')
	else:
		raise ValueError('Unknown likelihood distribution. The distribution needs to be Poisson, negative binomial or normal.')
			
def NB_sampledZGenerator(field_data, cali_iter, input_args_dict, nugget, const):
	# 1. if this is the first calibration run
	if cali_iter == 0:
		sampled_Z_Log = 0
		pdfMuLogGivenM_S = 0
		nb_dispersion_previous = 100
	else:
		sampled_Z_Log = input_args_dict['sampled_Z_log']
		pdfMuLogGivenM_S = input_args_dict['pdfMuLogGivenM_S']
		nb_dispersion_previous = input_args_dict['nb_dispersion']
	
	# 2. read-in other necessary input arguments
	param_t = input_args_dict['param_temp']
	param_t_aug = input_args_dict['param_temp_aug']
	
	train_inputs = input_args_dict['train_inputs']
	train_inputs_aug = input_args_dict['train_inputs_aug']
	train_outputs_log = input_args_dict['train_outputs_log']
	len_train_outputs = train_outputs_log.shape[0]
	
	emulator_param_t = input_args_dict['emulator_param_t']
	V_t = input_args_dict['V_t']
	
	rho = input_args_dict['rho']
	corrMat_train_inputs_inv = input_args_dict['corrMat_train_inputs_inv']
	
	field_tspan = input_args_dict['field_tspan']
	len_field_data = len(field_tspan)
	
	AR_order = input_args_dict['AR_order']
	sampleZ_iter_num = input_args_dict['sampleZ_iter_num']
	
	# 3. initialize and pre-allocate memory
	predMu_log = np.zeros((len_train_outputs, 1))
	predMu_log_star = np.zeros_like(predMu_log)
	predMu = np.zeros_like(predMu_log)
	predMu_star = np.zeros_like(predMu_log)
	
	log_pdf_z_given_dispersion = np.zeros((len_field_data))
	log_pdf_z_given_dispersion_star = np.zeros_like(log_pdf_z_given_dispersion)	
	pdfMuLogGivenMtSt = np.zeros_like(predMu)
	
	mt = np.zeros_like(predMu)
	st = np.zeros_like(predMu)	
	
	predMu_log_tPlus1 = np.zeros((len_train_outputs-2, 1))
	m_tPlus1 = np.zeros_like(predMu_log_tPlus1)
	m_star_tPlus1 = np.zeros_like(predMu_log_tPlus1)
	s_tPlus1 = np.zeros_like(predMu_log_tPlus1)
	
	'''
	3.2 Set the dispersion of negative binomial to the initial value, proposal
	variance of dispersion is set to 0.15, and the prior of dispersion is set
	to be dispersion ~ Gamma(0.01, 0.01), e.g. gampdf(X, 0.01, 0.01).
	
	In addition, we also set the variance of proposal for mu of gamma to 0.15.
	'''
	nb_dispersion = nb_dispersion_previous
	log_nb_dispersion = np.log(nb_dispersion_previous)
	sigma_dispersion_log = 0.15
	sigma_mu_log = 0.15
	
	# 3.3 initialize
	initVal = generateInitVal_weekly_noDailyData(param_t, train_inputs)
	initVal_log = np.log(initVal + nugget) - const
	
	# 3.4 update the initial elements in space
	predMu[0, :] = initVal
	predMu_log[0, :] = initVal_log
	mt[0, :] = predMu_log[0, :]
	
	# -------------------------------------------------------- #
	#                                                          #
	#                                                          #
	#                                                          #
	#                       Main Loop                          #
	#                                                          #
	#                                                          #
	#                                                          #
	# -------------------------------------------------------- #
	for b in range(sampleZ_iter_num):
		# ========== first calibration run + first sampling run ========== #
		if b == 0 and cali_iter == 0:
			# loop over time steps
			for t in range(AR_order, len_train_outputs):
				mt_para = np.dot(np.r_[predMu_log[t-AR_order, :].ravel(), param_t_aug.ravel()], emulator_param_t[:, t])
				mt_X = np.dot(np.c_[train_outputs_log[t-AR_order, :].T.ravel(), train_inputs_aug], emulator_param_t[:, t])
				
				mt[t, :] = mt_para + np.dot(np.dot(rho.T, corrMat_train_inputs_inv), (train_outputs_log[t, :].T - mt_X))
				
				st[t, :] = V_t[t] * (1 - np.dot(np.dot(rho.T, corrMat_train_inputs_inv), rho))
				
				st[t, :] = np.where(st[t, :] < 0, 0, st[t, :])
				
				predMu_log[t, :] = mt[t, :] + np.sqrt(st[t, :]) * np.random.randn(1)
				predMu[t, :] = np.exp(predMu_log[t, :] + const) - nugget
				pdfMuLogGivenMtSt[t, :] = norm.pdf(predMu_log[t, :], loc=mt[t, :], scale=np.sqrt(st[t, :]))
				
				
		# ========== 2+ calibration run + first sampling run ========== #
		elif b == 0 and cali_iter != 0:
			predMu_log[AR_order: , :] = sampled_Z_Log[AR_order: , :]
			predMu[AR_order:, :] = np.exp(predMu_log[AR_order: , :] + const) - nugget
			pdfMuLogGivenMtSt[AR_order:, :] = pdfMuLogGivenM_S[AR_order: , :]			
		
		# ========== 2+ calibration run + 2+ sampling run ========== #
		else:
			# ---- propose value for dispersion, then accept/rej the proposed value ---- #
			log_nb_dispersion_star = np.log(nb_dispersion) + sigma_dispersion_log * np.random.randn(1)
			nb_dispersion_star = np.exp(log_nb_dispersion_star)
			
			# calculate the p(z|dispersion): log-gamma
			for t in field_tspan:
				log_pdf_z_given_dispersion[t-4] = log_pdf_z_given_dispersion_fcn(field_data[t-4, :], predMu[t, :], nb_dispersion)
				log_pdf_z_given_dispersion_star[t-4] = log_pdf_z_given_dispersion_fcn(field_data[t-4, :], predMu[t, :], nb_dispersion_star)
				
			# calculate the prior
			dispersion_prior = gamma.pdf(nb_dispersion, a = 0.01, scale = 100)
			dispersion_star_prior = gamma.pdf(nb_dispersion_star, a = 0.01, scale = 100)
			
			# acceptance probability
			log_numerator_dispersion_acceptProb = np.log(dispersion_star_prior) + sum(log_pdf_z_given_dispersion_star) + log_nb_dispersion_star
			log_determinant_dispersion_acceptProb = np.log(dispersion_prior) + sum(log_pdf_z_given_dispersion) + log_nb_dispersion
			log_acceptProb = log_numerator_dispersion_acceptProb - log_determinant_dispersion_acceptProb
			
			if np.log(np.random.uniform(size = 1)) < log_acceptProb:
				nb_dispersion = nb_dispersion_star
				log_nb_dispersion = log_nb_dispersion_star
				log_pdf_z_given_dispersion = log_pdf_z_given_dispersion_star
				
			# ---- END propose value for dispersion, then accept/rej the proposed value ---- #
			
			# ---- propose value for mean of negative binomial ---- #
			for t in range(AR_order, len_train_outputs):
				mt_para = np.dot(np.r_[predMu_log[t-AR_order, :].ravel(), param_t_aug.ravel()], emulator_param_t[:, t])
				mt_X = np.dot(np.c_[train_outputs_log[t-AR_order, :].T.ravel(), train_inputs_aug], emulator_param_t[:, t])
				
				mt[t, :] = mt_para + np.dot(np.dot(rho.T, corrMat_train_inputs_inv), (train_outputs_log[t, :].T - mt_X))
				
				st[t, :] = V_t[t] * (1 - np.dot(np.dot(rho.T, corrMat_train_inputs_inv), rho))
				
				st[t, :] = np.where(st[t, :] < 0, 0, st[t, :])
				
				predMu_log[t, :] = predMu_log[t, :] + sigma_mu_log * np.random.randn(1)
				predMu_star[t, :] = np.exp(predMu_log_star[t, :] + const) - nugget
				
				# *********** we first deal with time stances t = 2:T-1, T will need the specical recipe
				if t < max(field_tspan):
					s_tPlus1[t-1, :] = V_t[t+1] * (1 - np.dot(np.dot(rho.T, corrMat_train_inputs_inv), rho))
					mt_para_tPlus1 = np.dot(np.r_[predMu_log[t, :].ravel(), param_t_aug], emulator_param_t[:, t+1])
					mt_X_tPlus1 = np.dot(np.c_[train_outputs_log[t, :].T.ravel(), train_inputs_aug], emulator_param_t[:, t+1])
					
					m_tPlus1[t-1, :] = mt_para_tPlus1 + np.dot(np.dot(rho.T, corrMat_train_inputs_inv), (train_outputs_log[t+1, :].T - mt_X_tPlus1))
					
					predMu_log_tPlus1[t-1, :] = predMu_log[t+1, :]
					
					mt_para_star_tPlus1 = np.dot(np.r_[predMu_log[t, :].ravel(), param_t_aug.ravel()], emulator_param_t[:, t+1])
					
					m_star_tPlus1[t-1, :] = mt_para_star_tPlus1 + np.dot(np.dot(rho.T, corrMat_train_inputs_inv), (train_outputs_log[t+1, :].T - mt_X_tPlus1))
				# *********** END: we first deal with time stances t = 2:T-1, T will need the specical recipe	
				
				# =========== calculate the log-acceptance ========= #
				if t != len_train_outputs-1:
					if t in [1, 2, 3]:
						# days 2, 3 and 4, which are not presented in field data
						accpRate = -0.5 * (predMu_log_star[t, :] - mt[t, :])**2 / st[t, :] - 0.5 * (predMu_log_tPlus1[t-1, :] - m_star_tPlus1[t-1, :])**2 / s_tPlus1[t-1, :]\
									+0.5 * (predMu_log[t, :] - mt[t, :])**2 / st[t, :] + 0.5 * (predMu_log_tPlus1[t-1, :] - m_tPlus1[t-1, :])**2 / s_tPlus1[t-1, :]
					else:
						# days: 5 - 46
						log_pdf_z_given_dispersion_star[t-4] = np.where(predMu_star[t, :] < 0, -np.infty, log_pdf_z_given_dispersion_fcn(field_data[t-4], predMu_star[t, :], nb_dispersion))
						accpRate = log_pdf_z_given_dispersion_star[t-4] - 0.5 * (predMu_log_star[t, :] - mt[t, :])**2 / st[t, :] - 0.5 * (predMu_log_tPlus1[t-1, :] - m_star_tPlus1[t-1, :])**2 / s_tPlus1[t-1, :]\
									-log_pdf_z_given_dispersion[t-4] + 0.5 * (predMu_log[t, :] - mt[t, :])**2 / st[t, :] + 0.5 * (predMu_log_tPlus1[t-1, :] - m_tPlus1[t-1, :])**2 / s_tPlus1[t-1, :]
				else:
					# day 47
					log_pdf_z_given_dispersion_star[t-4] = np.where(predMu_star[t, :] < 0, -np.infty, log_pdf_z_given_dispersion_fcn(field_data[t-4, :], predMu_star[t, :], nb_dispersion))
					accpRate = log_pdf_z_given_dispersion_star[t-4] - 0.5 * (predMu_log_star[t, :] - mt[t, :])**2 / st[t, :]\
								-log_pdf_z_given_dispersion[t-4] + 0.5 * (predMu_log[t, :] - mt[t, :])**2 / st[t, :]
				# =========== END: calculate the log-acceptance ========= #
							
				if np.log(np.random.uniform(size = 1)) < accpRate:
					predMu_log[t, :] = predMu_log_star[t, :]
					pdfMuLogGivenMtSt[t, :] = norm.pdf(predMu_log[t, :], loc=mt[t, :], scale=np.sqrt(st[t, :]))
					predMu[t, :] = predMu_star[t, :]
				
	# return the final resutls
	res_dict = {}
	res_dict['predMu'] = predMu
	res_dict['predMu_log'] = predMu_log
	res_dict['pdfMuLogGivenMtSt'] = pdfMuLogGivenMtSt
	res_dict['nb_dispersion'] = nb_dispersion
	return res_dict
	

		
	

def Pois_sampledZGenerator():
	res_dict
	return res_dict
	
def generateInitVal_weekly_noDailyData(param, X):
	# this is the function to generate the initial values of observation at time 0 given
	# the update model parameters (eta)
	max_X = X.max(axis = 0)
	min_X = X.min(axis = 0)
	
	factor_X = max_X - min_X
	
	param_og = (param * factor_X + min_X).ravel()
	
	# Population size
	population = sum([119967, 410926, 840309, 989473, 2747554, 1566445, 882256])
	
	# Population susceptibility
	susceptibility = 0.8903142
	
	# delay.distribution <- discretised.delay.cdf(delay.to.GP)[1:2]
	
	delay_distribution  = np.array([0.00356498122310219, 0.0241909650774591,\
	    0.0537030705484224,  0.07956071762201, 0.0961914714596036,\
	    0.102950833972866, 0.101591781393376,  0.0946065162134661,\
	    0.0843575216352006, 0.0727223345103803, 0.061022715024489,\
	    0.050087644508563, 0.0403632249361386, 0.0320249177166795])
	
	delay_distribution = np.flip(np.cumsum(delay_distribution), axis=0)
	
	# Main commands
	R = param_og[1] * param_og[2] * ((param_og[1] + 1)**2) / (1 - (1 / (((param_og[1] * param_og[2] / 2) + 1)**2)))
	rho = 1 / param_og[2]
	sigma = 0.5
	alpha = np.exp(param_og[1]/2) - 1
	I0 = np.exp(param_og[3]) * param_og[2] * population / (param_og[0] * R)
	
	I1 = []
	I1.append(I0 / (1 + (rho / (alpha + rho))))
	
	I2 = []
	I2.append(I1[-1] * (rho / (alpha + rho)))
	
	E2 = []
	E2.append(I1[-1] * ((alpha + rho) / sigma))
	
	E1 = []
	E1.append(E2[-1] * ((alpha + sigma) / sigma))
	
	p_beta = R / (param_og[2] * population)
	
	p_lambda = []
	p_lambda.append((I1[-1] + I2[-1]) * p_beta / 2)
	
	S = []
	S.append(population * susceptibility)
	
	nni = []
	nni.append(p_lambda[-1] * S[-1])
	
	for t in range(2, 15):
		S.append(S[-1] * (1 - p_lambda[-1]))
		E1.append((0.5 * E1[-1]) + (p_lambda[-1] * S[-2]))
		E2.append((0.5 * E2[-1]) + (0.5 * E1[-2]))
		I1.append(((1 - (1 / param_og[2])) * I1[-1]) + (0.5 * E2[-2]))
		I2.append(((1 - (1 / param_og[2])) * I2[-1]) + ((1 / param_og[2]) * I1[-2]))
		nni.append(p_lambda[-1] * S[-2])
		p_lambda.append((I1[-1] + I2[-1]) * p_beta / 2)
	
	nni = np.array(nni) * param_og[0]
	init_val_weekly_case = sum(nni * np.array(delay_distribution))
	
	return init_val_weekly_case

def log_pdf_z_given_dispersion_fcn(z_t, mu_t, dispersion_t):
	r = mu_t / dispersion_t
	p = dispersion_t / (dispersion_t + 1)
	
	log_pdf_z_dispersion = gamma.logpdf((z_t+r), a = 0.01, scale = 100) - \
							 gamma.logpdf(r, a = 0.01, scale = 100) +\
							 r * np.log(1-p) + z_t * np.log(p)
	
	return log_pdf_z_dispersion

def pdfMuLogGivenM_S_evaluate(param_aug, emulator_param_temp, V_temp, train_outputs_log, train_inputs_aug, rho_temp, field_tspan, corrMat_train_inputs_inv, AR_order, sampled_Z_log, weekly=True):
	pdfMuLogGivenM_S = np.zeros_like(sampled_Z_log)
	mt = np.zeros_like(sampled_Z_log)
	st = np.zeros_like(sampled_Z_log)
	
	len_train_outputs = train_outputs_log.shape[0]
	
	for t in range(AR_order, len_train_outputs):
		mt_para = np.dot(np.r_[sampled_Z_log[t-AR_order, :].ravel(), param_aug.ravel()], emulator_param_temp[:, t])
		mt_X = np.dot(np.c_[train_outputs_log[t-AR_order, :].T.ravel(), train_inputs_aug], emulator_param_temp[:, t])
		
		mt[t, :] = mt_para + np.dot(np.dot(rho_temp.T, corrMat_train_inputs_inv), (train_outputs_log[t, :].T - mt_X))
		
		st[t, :] = V_temp[t] * (1 - np.dot(np.dot(rho_temp.T, corrMat_train_inputs_inv), rho_temp))
		
		st[t, :] = np.where(st[t, :] < 0, 0, st[t, :])
		
		pdfMuLogGivenM_S[t, :] = norm.pdf(sampled_Z_log[t, :], loc=mt[t, :], scale=np.sqrt(st[t, :]))

	return pdfMuLogGivenM_S

def MT_fcn(sampled_Z_log, train_outputs_log, emulator_param, corrMat_train_inputs_inv, rho, train_inputs_aug, param_aug, AR_order):
	len_sampled_Z = sampled_Z_log.shape[0]
	out_res = np.zeros((len_sampled_Z - AR_order, 1))
	
	for t in range(AR_order, len_sampled_Z):
		mz = np.dot(np.r_[sampled_Z_log[t-AR_order, :].ravel(), param_aug.ravel()], emulator_param[:, t])
		my = np.dot(np.c_[train_outputs_log[t-AR_order, :].T.ravel(), train_inputs_aug], emulator_param[:, t])
		out_res[t-AR_order, :] = mz + np.dot(np.dot(rho.T, corrMat_train_inputs_inv), (train_outputs_log[t, :].T - my))
	return out_res

def ST_fcn(sampled_Z_log, V, corrMat_train_inputs_inv, rho, AR_order):
	len_sampled_Z = sampled_Z_log.shape[0]
	out_res = np.zeros((len_sampled_Z-AR_order, 1))
	
	for t in range(AR_order, len_sampled_Z):
		out_res[t-1, :] = V[t] * (1 - np.dot(np.dot(rho.T, corrMat_train_inputs_inv), rho))
	return out_res

if __name__ == '__main__':
	print('ok')

















































