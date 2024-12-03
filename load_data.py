import os
import torch
import numpy as np

from tqdm.auto import tqdm
from torch.utils.data import Dataset


def generateZipf_nips(n, a=1):
    '''
    Create data corresponding to a Zipf distribution
    over n elements with freq n/i^a for element i
    
    Calculate the error of the sketch
    '''
    a = np.array([n//(i**a) for i in range(1, n+1)])
    s = np.random.permutation(len(a))
    return a[s].tolist()


def readTraces(path, name='', KEY_T_SIZE=8, num=2_000_000, skewness=1.01):
    assert os.path.isfile(path) or name == 'synthetic', "File not found"
    traces = []
    
    if name == 'network':
        print("Reading in packets data...")

        with open(path, 'rb') as input_data:
            while True:
                str_data = input_data.read(KEY_T_SIZE)
                if len(str_data) < KEY_T_SIZE:
                    break
                traces.append(str_data)

    elif name == 'synthetic':
        # data = generateZipf(num)
        data = np.random.zipf(skewness, size=num).tolist()
        traces += [i.to_bytes(KEY_T_SIZE, 'little') for i in data]

    else:
        with open(path, 'r') as file:
            data = file.readlines()

        with tqdm(initial=0, total=len(data), desc='Reading in data') as pbar:
            for line in data:
                L = list(map(int, line.strip().split()))
                traces += [i.to_bytes(KEY_T_SIZE, 'little') for i in L]
                pbar.update(1)
    
    size = len(traces)
    print(f'Successfully read in {size} items.')
    return size, traces


class sketchDataset(Dataset):
    def __init__(self, data):
        super(sketchDataset, self).__init__()
        self.sample_num = data.shape[0]
        self.samples = data

    def __getitem__(self, ind):
        return torch.from_numpy(self.samples[ind, :, :]).float()
    
    def __len__(self):
        return self.sample_num


if __name__ == '__main__':
    # path = 'Source_Data/kosarak/kosarak.dat'
    # path = 'Source_Data/retail/retail.dat'
    path = 'Source_Data/caida/test-8s.dat'
    traces = readTraces(path, name='caida')
