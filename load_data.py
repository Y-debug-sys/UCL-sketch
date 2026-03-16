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
    
    Args:
        n: Number of elements in the distribution
        a: Exponent parameter for the Zipf distribution (default 1)
        
    Returns:
        A randomly permuted list of frequencies following Zipf distribution
    '''
    a = np.array([n//(i**a) for i in range(1, n+1)])
    s = np.random.permutation(len(a))
    return a[s].tolist()


def readTraces(path, name='', KEY_T_SIZE=8, num=2_000_000, skewness=1.01):
    """
    Read trace data from various sources (real or synthetic) and convert to byte format
    
    Args:
        path: Path to the data file (ignored for synthetic data)
        name: Type of data ('network', 'retail', 'kosarak', 'synthetic')
        KEY_T_SIZE: Size of each key in bytes
        num: Number of items to generate for synthetic data
        skewness: Skewness parameter for Zipf distribution (for synthetic data)
        
    Returns:
        Tuple of (size, traces) where size is the number of items and traces is the list of trace data
    """
    assert os.path.isfile(path) or name == 'synthetic', "File not found"
    traces = []
    
    if name == 'network':
        # Handle network packet data - read fixed-size byte chunks
        print("Reading in packets data...")

        with open(path, 'rb') as input_data:
            while True:
                str_data = input_data.read(KEY_T_SIZE)
                if len(str_data) < KEY_T_SIZE:
                    break
                traces.append(str_data)

    elif name == 'synthetic':
        # Generate synthetic data following Zipf distribution
        # data = generateZipf(num)
        data = np.random.zipf(skewness, size=num).tolist()
        traces += [i.to_bytes(KEY_T_SIZE, 'little') for i in data]

    else:
        # Handle transaction data (retail, kosarak) - read integers from file
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
    """
    Custom Dataset class for sketch data to be used with PyTorch DataLoader
    This wraps the sketch data for efficient batching during training
    """
    def __init__(self, data):
        super(sketchDataset, self).__init__()
        self.sample_num = data.shape[0]  # Number of samples in the dataset
        self.samples = data              # Actual sketch data

    def __getitem__(self, ind):
        """
        Get a specific sample from the dataset
        
        Args:
            ind: Index of the sample to retrieve
            
        Returns:
            The sample at the given index as a PyTorch tensor
        """
        return torch.from_numpy(self.samples[ind, :, :]).float()
    
    def __len__(self):
        """
        Get the total number of samples in the dataset
        
        Returns:
            Number of samples in the dataset
        """
        return self.sample_num