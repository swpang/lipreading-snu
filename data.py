from collections import Iterator, defaultdict
from torch.utils.data import DataLoader
from datasets import DATASET

# =====================
# Base Classes and ABCs
# =====================


class DataScheduler(Iterator):
	def __init__(self, config):
		self.config = config
		self.dataset = DATASET[self.config['dataset']](config, mode='train')
		self.eval_datasets = [
			DATASET[x[0]](config, mode=x[1])
			for x in self.config['eval_datasets']
		]

		if self.config.get('test_datasets') is not None:
			self.test_datasets = [
				DATASET[x[0]](config, mode=x[1])
				for x in self.config['test_datasets']
			]
		self.total_epoch = self.config['epoch']
		self.step_cnt = 0
		self.epoch_cnt = 0
		self._remainder = len(self.dataset)
		self.data_loader = DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			shuffle=True
		)
		self.iter = iter(self.data_loader)

	def __next__(self):
		'''
		:return:
			data: dict of corresponding data
		'''
		if self.data_loader is None:
			raise StopIteration
		try:
			data = next(self.iter)
		except StopIteration:
			self.iter = iter(self.data_loader)
			data = next(self.iter)
		self.update_epoch_cnt()
		self.step_cnt += 1
		return data, self.epoch_cnt

	def __len__(self):
		return len(self.sampler)

	def check_eval_step(self, step):
		if (step + 1) < self.config['min_eval_step']:
			return False
		return ((step + 1) % self.config['eval_step'] == 0) \
			   or self.config['debug_eval']

	def check_test_step(self, step):
		if (step + 1) < self.config['min_test_step']:
			return False

		return (step + 1) % self.config['test_step'] == 0 \
			if self.config.get('test_step') is not None else False

	def check_summary_step(self, step):
		return (step + 1) % self.config['summary_step'] == 0

	def evaluate(self, model, writer, step):
		for eval_dataset in self.eval_datasets:
			eval_dataset.evaluate(model, writer, step)

	def test(self, model, writer, step):
		print('Testing...')
		if self.test_datasets is not None:
			for test_dataset in self.test_datasets:
				test_dataset.test(model, writer, step)

	def update_epoch_cnt(self):
		self._remainder -= self.config['batch_size']
		if self._remainder < self.config['batch_size']:
			self._remainder += len(self.dataset)
			self.epoch_cnt += 1

