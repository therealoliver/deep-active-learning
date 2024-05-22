import argparse
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy
from pprint import pprint
from multiprocessing import Pool

def active_train(strategy_name, n_init_labeled):
	print(f'train for {strategy_name}: {n_init_labeled}')
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1, help="random seed")
	parser.add_argument('--n_init_labeled', type=int, default=n_init_labeled, help="number of init labeled samples")
	parser.add_argument('--n_query', type=int, default=1000, help="number of queries per round")
	parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
	parser.add_argument('--dataset_name', type=str, default="CIFAR10", help="dataset")
	parser.add_argument('--strategy_name', type=str, default=strategy_name, help="query strategy")

	args = parser.parse_args()
	pprint(vars(args))

	with open(strategy_name + f'-{n_init_labeled}.txt', 'a') as f:
		f.write(f'{n_init_labeled}\n')

	# fix random seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.enabled = False

	# device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('run on {}'.format(device))

	dataset = get_dataset(args.dataset_name)  # load dataset
	net = get_net(args.dataset_name, device)  # load network
	strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

	# start experiment
	dataset.initialize_labels(args.n_init_labeled)
	print(f"number of labeled pool: {args.n_init_labeled}")
	print(f"number of unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
	print(f"number of testing pool: {dataset.n_test}")
	print()

	# round 0 accuracy
	print("Round 0")
	max_acc = strategy.train()
	preds = strategy.predict(dataset.get_test_data())
	print(f"Round 0 test acc: {dataset.cal_test_acc(preds)}, max acc: {max_acc}")

	with open(strategy_name + f'-{n_init_labeled}.txt', 'a') as f:
		f.write(f'{max_acc}\t')

	for rd in range(1, args.n_round + 1):
		print(f"Round {rd}")

		# query
		query_idxs = strategy.query(args.n_query)

		# update labels
		strategy.update(query_idxs)
		max_acc = strategy.train()

		# calculate accuracy
		preds = strategy.predict(dataset.get_test_data())
		print(f"Round {rd} test acc: {dataset.cal_test_acc(preds)}, max acc: {max_acc}")

		with open(strategy_name + f'-{n_init_labeled}.txt', 'a') as f:
			f.write(f'{max_acc}\t')

	with open(strategy_name + f'-{n_init_labeled}.txt', 'a') as f:
		f.write('\n')


def full_train(n_init_labeled):
	print(f'train for full sample: {n_init_labeled}')
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=1, help="random seed")
	parser.add_argument('--n_init_labeled', type=int, default=n_init_labeled, help="number of init labeled samples")
	parser.add_argument('--dataset_name', type=str, default="CIFAR10", help="dataset")
	parser.add_argument('--strategy_name', type=str, default="RandomSampling", help="query strategy")

	args = parser.parse_args()
	pprint(vars(args))

	# fix random seed
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.backends.cudnn.enabled = False

	# device
	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print('run on {}'.format(device))

	dataset = get_dataset(args.dataset_name)  # load dataset
	net = get_net(args.dataset_name, device)  # load network
	strategy = get_strategy(args.strategy_name)(dataset, net)  # load strategy

	# start experiment
	dataset.initialize_labels(args.n_init_labeled)
	print(f"number of labeled pool: {args.n_init_labeled}")
	print(f"number of unlabeled pool: {dataset.n_pool - args.n_init_labeled}")
	print(f"number of testing pool: {dataset.n_test}")
	print()

	# accuracy
	max_acc = strategy.train()
	preds = strategy.predict(dataset.get_test_data())
	print(f"test acc: {dataset.cal_test_acc(preds)}, max acc: {max_acc}")

	with open(f'full_train-{n_init_labeled}.txt', 'a') as f:
		f.write(f'{max_acc}\n')


def call_back(res):
	print(res)


def err_call_back(err):
	print(f'error：{str(err)}')


if __name__ == '__main__':
	pool = Pool()

	# 增量学习
	for strategy_name in ["RandomSampling", "LeastConfidence", "MarginSampling", "EntropySampling"]:
		for n_init_labeled in [1000, 2000, 5000, 10000]:
			pool.apply_async(active_train, args=(strategy_name, n_init_labeled), error_callback=err_call_back, callback=call_back)

	# 全量学习
	for n_init_labeled in range(1000, 21000, 1000):
		pool.apply_async(full_train, args=(n_init_labeled,), error_callback=err_call_back, callback=call_back)
