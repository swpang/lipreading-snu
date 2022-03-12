import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from models.base_model import Model
from torch.utils.tensorboard import SummaryWriter
from typing import List


class BaseDataset(Dataset, ABC):
	name = 'base'

	def __init__(self, config: dict, mode: str = 'train'):
		self.config = config
		self.mode = mode
		self.device = config['device']
		self.summary_name = self.name

	def __getitem__(self, idx) \
			-> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, List[str]):
		# sparse tensor and tensor should have equal size
		raise NotImplemented

	def __iter__(self):
		while True:
			idx = random.randint(0, len(self) - 1)
			yield self[idx]

	def collate_fn(self, batch: List) -> dict:
		# convert list of dict to dict of list
		raise NotImplemented

	def evaluate(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()
		data_loader = DataLoader(
			self,
			batch_size=self.config['eval_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
		)

		print('')
		eval_corrects = 0
		eval_losses = []

		for eval_step, data in enumerate(data_loader):
			eval_correct, eval_loss = model.evaluate(data, step, self.mode)
			eval_corrects += eval_correct
			eval_losses.append(eval_loss)

			print('\r[Evaluating, Step {:7}, Loss {:5}, Correct {:5}]'.format(
					eval_step, '%.3f' % eval_loss, eval_correct
				), end=''
			)
		print('\r\n[Evaluated, Loss {:5}, Accuracy {:5}]'.format(
					'%.3f' % np.mean(np.array(eval_losses)), eval_corrects/25000
				), end=''
			)
		model.scalar_summaries['loss/{}/total'.format(self.mode)] \
			+= [np.mean(np.array(eval_losses))]

		model.scalar_summaries['accuracy/{}/total'.format(self.mode)] \
			+= [eval_corrects/25000]

		print('')
		model.write_dict_summaries(step)
		model.train(training)

	def test(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()
		data_loader = DataLoader(
			self,
			batch_size=self.config['test_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
		)

		print('')
		corrects = 0
		losses = []

		for test_step, data in enumerate(data_loader):
			len_data = len(data['input'])
			correct, loss = model.evaluate(data, step, self.mode)
			corrects += correct
			losses.append(loss / len_data)
			print('\r[Testing, Step {:7}, Loss {:5}, Correct {:5}]'.format(
				test_step, '%.3f' % (loss / len_data), correct
			), end=''
			)
		print('\r\n[Tested, Loss {:5}, Accuracy {:5}]'.format(
			'%.3f' % np.mean(np.array(losses)), corrects / 25000
		), end=''
		)
		model.scalar_summaries['loss/{}/total'.format(self.mode)] \
			+= [np.mean(np.array(losses))]

		model.scalar_summaries['accuracy/{}/total'.format(self.mode)] \
			+= [corrects / 25000]

		print('')
		model.write_dict_summaries(step)
		model.train(training)
