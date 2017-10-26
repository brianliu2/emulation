# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 16:15:48 2017

@author: xliu
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20

def read_data_from_txt(dataName, is_output, time_length = None, data_type = np.float64):
	data_list = []
	with open(dataName) as file:
	    for line in file:
	        line_context = [t for t in line.split('\n')[0].strip().split(' ') if t != '']
	        data_list.append(line_context)
	
	data = np.array(data_list, dtype = data_type)
	
	if is_output:
		tspan = range(time_length)
		data  = data.T[tspan, :]
	return data

def input_normalize_denormalize(inputs, is_normalize, normalize_inputs = None):
	if is_normalize:
		inputs_norm = (inputs-inputs.min(axis=0)) / (inputs.max(axis=0) - inputs.min(axis=0))
		return inputs_norm
	else:
		maxInputs = inputs.max(axis = 0)
		minInputs = inputs.min(axis = 0)
		minInputs_lst = list(minInputs)
		minFactor = np.array([minInputs_lst, ]*normalize_inputs.shape[0])
		reScaleFactor = list(maxInputs-minInputs)
		reScaleFactor = np.array([reScaleFactor, ]*normalize_inputs.shape[0])
		inputs_denormalize = normalize_inputs * reScaleFactor + minFactor
		return inputs_denormalize

def log_trans(outpus, nugget, const):
	outputs_log = np.log(outpus + nugget) - const
	return outputs_log

def input_augment(inputs):
	num_inputs = inputs.shape[0]
	if num_inputs > 1:
		input_aug_res = np.hstack((np.ones((num_inputs, 1)), inputs))
	elif num_inputs == 1:
		input_aug_res = np.append(1, inputs)
	else:
		print('Error: number of input points must be a positive integer.')
	return input_aug_res

if __name__ == '__main__':
	data_path = os.path.join('/Users/xliu/Documents/MRC/', 
							   'Work/Program/emulator/marian/JASA_2014_code/')
	train_input  = data_path + 'LHCDesign_training.txt'
	train_output = data_path + 'Outputs_training.txt'
	valid_input  = data_path + 'LHCDesign_validation.txt'
	valid_output = data_path + 'Outputs_validation.txt'

	train_input  = read_data_from_txt(train_input, is_output = False)
	train_output = read_data_from_txt(train_output, is_output = True, time_length = 245)
	valid_input  = read_data_from_txt(valid_input, is_output = False)
	valid_output = read_data_from_txt(valid_output, is_output = True, time_length = 245)
	
	train_inputs_norm = input_normalize_denormalize(train_input, is_normalize=True)
	
	train_output_log = log_trans(train_output, nugget=0.5, const=5)
#	plt.figure(figsize=(16, 8))
#	plt.subplot(1, 2, 1)
#	plt.plot(train_output)
#	plt.title('$\mu$($\eta$, t)')
#	plt.xlabel('time', fontsize = 20)
#	plt.xlim([0, 245]);
#	
#	plt.subplot(1, 2, 2)
#	plt.plot(train_output_log)
#	plt.title('log $\mu$($\eta$, t)')
#	plt.xlabel('time', fontsize = 20)
#	plt.xlim([0, 245]);
	
	train_input_aug = input_augment(train_inputs_norm)
	print(train_input_aug.shape)



























































