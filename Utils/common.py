import math
import torch
import numpy as np


def isPrime(num: int):
    """
    Check if a number is prime (has no divisors other than 1 and itself)
    
    Args:
        num: Integer to check for primality
        
    Returns:
        True if the number is prime, False otherwise
    """
    border = int(math.sqrt(float(num)))
    for i in range(2, border):
        if num % i == 0:
            return False
    return True


def calNextPrime(num: int):
    """
    Find the smallest prime number greater than or equal to the input number
    
    Args:
        num: Starting integer to find the next prime from
        
    Returns:
        The next prime number >= num
    """
    while not isPrime(num):
        num += 1
    return num


class Model_Args(object):
    """
    Container class for model hyperparameters and configuration settings
    Used to pass arguments to the neural network and training components
    """
    def __init__(
        self,
        n_extra_layers,
        d_model,
        dropout,
        share_dim,
        patience,
        batch_size,
        num_workers,
        learning_rate,
        train_epochs
    ):
        # Model architecture parameters
        self.e_layers = n_extra_layers  # Number of extra layers in the model
        self.d_model = d_model          # Model dimension
        self.dropout = dropout          # Dropout rate for regularization
        self.d_share = share_dim        # Shared dimension for parameter efficiency
        # Training parameters
        self.patience = patience        # Patience for early stopping
        self.batch_size = batch_size    # Batch size for training
        self.num_workers = num_workers  # Number of workers for data loading
        self.learning_rate = learning_rate  # Learning rate for optimization
        self.train_epochs = train_epochs    # Maximum number of training epochs

    def update_size(self, width, depth):
        """
        Update the sketch dimensions for the model
        
        Args:
            width: Width of the Count-Min sketch (bucket dimension)
            depth: Depth of the Count-Min sketch (hash function dimension)
        """
        self.bucket_dim = width    # Bucket dimension (width of CM sketch)
        self.hash_num = depth      # Number of hash functions (depth of CM sketch)

    def update_path(self, ckpt_path):
        """
        Update the checkpoint path for saving/loading models
        
        Args:
            ckpt_path: Directory path to save/load model checkpoints
        """
        self.checkpoints = ckpt_path

    def update_interval(self, interval):
        """
        Update the sampling interval for collecting training samples
        
        Args:
            interval: Number of insertions between consecutive samples
        """
        self.interval = interval

    def update_gpu(self, use_gpu, use_multi_gpu=False, gpu_id=0):
        """
        Update GPU configuration settings
        
        Args:
            use_gpu: Whether to use GPU for training/inference
            use_multi_gpu: Whether to use multiple GPUs
            gpu_id: ID of the GPU to use if not using multiple GPUs
        """
        self.use_multi_gpu = use_multi_gpu
        self.use_gpu = use_gpu
        self.gpu = gpu_id

    def select_ablation(self, ablation_type):
        """
        Select the ablation study type for the model
        
        Args:
            ablation_type: Type of ablation to perform (0 for standard, others for variants)
        """
        self.ablation = ablation_type


class Exp_Basic(object):
    """
    Basic experimental class providing common functionality for training and inference
    Defines the basic structure for experiments in the UCL-sketch framework
    """
    def __init__(self, args):
        self.args = args           # Arguments container
        self.device = self._acquire_device()  # Device (CPU/GPU) to run on
        self.model = self._build_model().to(self.device)  # Neural network model
    def _build_model(self):
        """Build the neural network model - to be implemented by subclasses"""
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        """Acquire the appropriate device (CPU or GPU) based on configuration"""
        if self.args.use_gpu:
            #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        """Load and prepare data - to be implemented by subclasses"""
        pass

    def train(self):
        """Train the model - to be implemented by subclasses"""
        pass

    def test(self):
        """Test the model - to be implemented by subclasses"""
        pass
    

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting during training
    Monitors validation loss and stops training when improvement stalls
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopping mechanism
        
        Args:
            patience: Number of epochs with no improvement after which training will stop
            verbose: Whether to print messages when a new best is found
            delta: Minimum change to qualify as an improvement
        """
        self.patience = patience      # Patience threshold
        self.verbose = verbose        # Verbosity flag
        self.counter = 0              # Counter for epochs without improvement
        self.best_score = None        # Best validation score observed so far
        self.early_stop = False       # Flag indicating if early stopping has been triggered
        self.val_loss_min = np.Inf    # Minimum validation loss observed so far
        self.delta = delta            # Minimum change to qualify as improvement

    def __call__(self, val_loss, model, path):
        """
        Call the early stopping mechanism during training
        
        Args:
            val_loss: Current validation loss
            model: Current model to potentially save
            path: Path to save the model checkpoint
        """
        score = -val_loss  # Negative because lower loss is better
        if self.best_score is None:
            # First iteration, set the baseline
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score - self.delta * self.best_score:
            # No significant improvement, increment counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping
        else:
            # Improvement detected, reset counter and save model
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Save the model checkpoint when a new best is achieved
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            path: Path to save the checkpoint
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pt')
        self.val_loss_min = val_loss


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate according to a predefined schedule
    
    Args:
        optimizer: PyTorch optimizer whose learning rate needs adjustment
        epoch: Current training epoch
        args: Arguments containing learning rate adjustment parameters
    """
    if args.lradj=='type1':
        # Type 1 schedule: reduce LR by half at epochs 100 and 150
        lr_adjust = {100: args.learning_rate * 0.5 ** 1, 150: args.learning_rate * 0.5 ** 2}
    elif args.lradj=='type2':
        # Type 2 schedule: more aggressive decay over multiple epochs
        lr_adjust = {50: args.learning_rate * 0.5 ** 1, 100: args.learning_rate * 0.5 ** 2,
                     150: args.learning_rate * 0.5 ** 3, 200: args.learning_rate * 0.5 ** 4,
                     250: args.learning_rate * 0.5 ** 5}
    else:
        # No adjustment schedule
        lr_adjust = {}
    if epoch in lr_adjust.keys():
        # Apply the learning rate adjustment
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))