import numpy as np
import pandas as pd
import os

import torch
import torch.utils.data as Data

from einops import rearrange

path = '/home/hjl/deep_learning_workspace/data_metrla/7212'

class trainDataset_metr_la(Data.Dataset):
	def __init__(self, transform=None, train=True):
		self.train_data = np.load(os.path.join(path, 'train.npz'), allow_pickle=True)
		self.x = self.train_data['x']
		self.y = self.train_data['y']
		self.y = self.y[:, :, :, 0]
		self.loc_npz = np.load(os.path.join(path, 'loc.npz'), allow_pickle=True)
		self.loc = self.loc_npz['arr_0']


	def __getitem__(self, index):
		x = torch.FloatTensor(self.x[index])
		x = rearrange(x, 't n d -> d n t')
		y = torch.FloatTensor(self.y[index])
		# y = y.transpose(0,1)
		loc = torch.FloatTensor(self.loc)

		return [x,y,loc]

	def __len__(self):
		return self.x.shape[0]

class valDataset_metr_la(Data.Dataset):
	def __init__(self, transform=None, train=True):
		self.val_data = np.load(os.path.join(path, 'val.npz'), allow_pickle=True)
		self.x = self.val_data['x']
		self.y = self.val_data['y']
		self.y = self.y[:, :, :, 0]
		self.loc_npz = np.load(os.path.join(path, 'loc.npz'), allow_pickle=True)
		self.loc = self.loc_npz['arr_0']

	def __getitem__(self, index):
		x = torch.FloatTensor(self.x[index])
		x = rearrange(x, 't n d -> d n t')
		y = torch.FloatTensor(self.y[index])
		# y = y.transpose(0,1)
		loc = torch.FloatTensor(self.loc)

		return [x,y, loc]

	def __len__(self):
		return self.x.shape[0]

class testDataset_metr_la(Data.Dataset):
	def __init__(self, transform=None, train=True):
		self.test_data = np.load(os.path.join(path, 'test.npz'), allow_pickle=True)
		self.x = self.test_data['x']
		self.y = self.test_data['y']
		self.y = self.y[:, :, :, 0]
		self.loc_npz = np.load(os.path.join(path, 'loc.npz'), allow_pickle=True)
		self.loc = self.loc_npz['arr_0']


	def __getitem__(self, index):
		x = torch.FloatTensor(self.x[index])
		x = rearrange(x, 't n d -> d n t')
		y = torch.FloatTensor(self.y[index])
		# y = y.transpose(0,1)
		loc = torch.FloatTensor(self.loc)

		return [x,y,loc]

	def __len__(self):
		return self.x.shape[0]
