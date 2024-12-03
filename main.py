import yaml
import torch
import random
import argparse
import importlib
import numpy as np

from tqdm.auto import tqdm
from Utils.mertrics import *
from load_data import readTraces
from Utils.training import learningSolver
from Utils.common import Model_Args


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def parse_args():
    parser = argparse.ArgumentParser(description='Main Experiment')
    
    parser.add_argument('--data', type=str, default='network', help='data',
                         choices=['retail', 'kosarak', 'network', 'synthetic'])
    parser.add_argument('--config_path', default=None, help='config file')
    parser.add_argument('--data_path', default=None, help='data file')
    parser.add_argument('--skewness', default=None, help='zipfian skewness')
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    
    parser.add_argument('--break_number', type=int, default=1000000, help='length of stream data')
    parser.add_argument('--ckpt', type=str, default='./checkpoints', help='location to store model checkpoints')
    
    # parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
    
    parser.add_argument('--save_pred', action='store_true', help='whether to save the estimated frequency', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
    
    parser.add_argument('--interval', type=int, default=1000, help='sampling inserval')
    parser.add_argument('--num_samples', type=int, default=128, help='maintained samples (sliding window)')
    parser.add_argument('--ablation', type=int, default=0, help='ablational type')

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    args = parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(args.gpu)

    config = load_yaml_config(args.config_path)

    print(f"Global seed set to {args.seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ucl_sketch = instantiate_from_config(config['sketch'])
    ucl_sketch.get_memory_usage()

    if args.data_path:
        size, traces = readTraces(args.data_path, args.data, config['sketch']['params']['KEY_T_SIZE'])
    elif args.skewness:
        assert (args.skewness and args.data=='synthetic'), 'Synthetic ?'
        size, traces = readTraces(args.data_path, args.data, config['sketch']['params']['KEY_T_SIZE'])

    if size > args.break_number:
        traces = traces[:args.break_number]
        size = args.break_number

    packetCnt = 0
    ground_truth = {}
    sample_initial = size - (args.interval + 12) * args.num_samples

    samples = np.empty([0, ucl_sketch.cm.depth, ucl_sketch.cm.width])

    with tqdm(initial=0, total=size, desc='Inserting packets into the sketch') as pbar:
        for idx, trace in enumerate(traces):
            if trace in ground_truth:
                ground_truth[trace] += 1
            else:
                ground_truth[trace] = 1
            
            ucl_sketch.insert(trace)
            packetCnt += 1
            pbar.update(1)
        
            if idx > sample_initial and idx % args.interval == 0:
                sample = ucl_sketch.get_current_state(return_A=False)
                samples = np.row_stack([samples, sample])

    print(f'Insert {packetCnt} items with {len(ground_truth)} distinct keys. Meanwhile, {samples.shape[0]} points is sampled.')

    A, index = ucl_sketch.get_current_state(return_A=True)
    model_args = instantiate_from_config(config['model'])
    model_args.update_size(ucl_sketch.cm.width, ucl_sketch.cm.depth)
    model_args.update_path(args.ckpt)
    model_args.update_interval(args.interval)
    model_args.update_gpu(args.use_gpu, args.use_multi_gpu, args.gpu)
    model_args.select_ablation(args.ablation)
    solver = learningSolver(model_args, A.shape[1])
    solver.train(samples, A, index)

    results = {'GT': [], 'UCL': []}

    print('Querying for all keys ...')

    for key, value in ground_truth.items():

        if ucl_sketch.cmResult == {}:
            test_sample = ucl_sketch.get_current_state(return_A=False)
            x = solver.test(test_sample)
            x = np.ceil(x.squeeze()).astype(np.int32)

        ucl_ans = ucl_sketch.query(key, x)
        results['GT'].append(value)
        results['UCL'].append(ucl_ans)

    GT = results['GT']
    ET = results['UCL']
    
    print('Performace of UCL-sketch:')
    AAE = average_absolute_error(GT, ET)
    ARE = average_relative_error(GT, ET)
    WMRD = weighted_mean_relative_difference(GT, ET)
    EAE = entropy_absolute_error(GT, ET)
    print(f'—— AAE: {AAE:.4f}, ARE: {ARE:.4f}, WMRD: {WMRD:.4f}, and Entropy Absolute Error: {EAE:.4f}')
