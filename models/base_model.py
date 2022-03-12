import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
from time import time
from typing import List


# ==========
# Model ABCs
# ==========


class Model(nn.Module, ABC):
	def __init__(self, config, writer: SummaryWriter):
		nn.Module.__init__(self)
		self.config = config
		self.device = config['device']
		self.writer = writer
		self.step_time = time()
		self.scalar_summaries = defaultdict(list)
		self.list_summaries = defaultdict(list)

	def get_lr(self):
		for param_group in self.optimizer.param_groups:
			return param_group['lr']

	def write_dict_summaries(self, step):

		# write scalar summaries
		for (k, v) in self.scalar_summaries.items():
			v = np.array(v).mean().item()
			self.writer.add_scalar(k, v, step)

		# write list summaries
		for (k, v) in self.list_summaries.items():
			self.writer.add_histogram(k, np.array(v), step)
		# reset summaries
		self.scalar_summaries.clear()
		self.list_summaries.clear()

	def write_summary(self, scheduler, step):
		# write summaries
		# write all the averaged summaries
		self.write_dict_summaries(step)

		# write current learning rate
		self.writer.add_scalar('lr', self.get_lr(), step)

		# write time elapsed since summary_step
		resource_prefix = 'resources/'
		self.writer.add_scalar(
			resource_prefix + 'time_per_step',
			(time() - self.step_time) / self.config['summary_step'], step
		)
		self.step_time = time()


	@abstractmethod
	def forward(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def learn(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def evaluate(self, data: dict, step: int, mode: str):
		raise NotImplementedError
