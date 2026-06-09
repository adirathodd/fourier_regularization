import torch
import einops
from math import gcd

class ModularAdditionDataset:
    """Dataset generator for modular addition problems."""
    
    def __init__(self, data_cfg):
        self.modulo = data_cfg['modulo'] # the numbed to divide by
        self.train_frac = data_cfg['train_frac'] # how much data to use for training
        self.data_seed = data_cfg['data_seed'] # for repeatable results 
        self.operation = data_cfg['operation']
        
        # Generate the complete dataset
        self.dataset, self.labels = self._generate_dataset()
        
        # Split into train/test
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._split_dataset()
    
    def _generate_dataset(self):
        """Generate all possible modular addition pairs."""
        # Create all combinations of a + b = c (mod MODULO)
        a_vector = einops.repeat(torch.arange(self.modulo), "i -> (i j)", j=self.modulo) # vectors a and b cover all possible pairs. ex: [0,0],[0,1],[1,2] ...
        b_vector = einops.repeat(torch.arange(self.modulo), "j -> (i j)", i=self.modulo)
        equals_vector = einops.repeat(torch.tensor(self.modulo), " -> (i j)", i=self.modulo, j=self.modulo)
        
        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)
        labels = (dataset[:, 0] + dataset[:, 1]) % self.modulo
        
        return dataset, labels

    def _generate_dataset_padded(self, seq_len=10):
        a_values = torch.arange(self.modulo)
        b_values = torch.arange(self.modulo)
        a_vector = einops.repeat(a_values, "i -> (i j)", j=self.modulo)
        b_vector = einops.repeat(b_values, "j -> (i j)", i=self.modulo)
        labels = (a_vector + b_vector) % self.modulo
        p_token = self.modulo
        dataset = []
        for a, b in zip(a_vector, b_vector):
            seq = torch.zeros(seq_len, dtype=torch.long)
            seq[2] = a
            seq[5] = b
            seq[-1] = p_token
            dataset.append(seq)
        dataset = torch.stack(dataset)
        return dataset, labels



    
    def _split_dataset(self):
        """Split dataset into train and test sets."""
        torch.manual_seed(self.data_seed)
        
        indices = torch.randperm(self.modulo * self.modulo)
        cutoff = int(self.modulo * self.modulo * self.train_frac)
        
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
        
        train_data = self.dataset[train_indices]
        train_labels = self.labels[train_indices]
        test_data = self.dataset[test_indices]
        test_labels = self.labels[test_indices]
        
        return train_data, train_labels, test_data, test_labels
    
    def to_device(self, device):
        """Move all data to specified device."""
        self.dataset = self.dataset.to(device)
        self.labels = self.labels.to(device)
        self.train_data = self.train_data.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_data = self.test_data.to(device)
        self.test_labels = self.test_labels.to(device)
        return self
    
    def get_train_data(self):
        """Get training data and labels."""
        return self.train_data, self.train_labels
    
    def get_test_data(self):
        """Get test data and labels."""
        return self.test_data, self.test_labels
    
    def get_full_data(self):
        """Get complete dataset and labels."""
        return self.dataset, self.labels
    
    def get_data_info(self):
        """Get information about the dataset."""
        return {
            'modulo': self.modulo,
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_frac': self.train_frac,
            'vocab_size': self.modulo + 1  # +1 for the equals token
        }
    
    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the dataset for inspection."""
        if subset == 'train':
            data, labels = self.train_data, self.train_labels
        elif subset == 'test':
            data, labels = self.test_data, self.test_labels
        else:
            data, labels = self.dataset, self.labels
        
        if idx is None:
            idx = 0
        
        sample_data = data[idx]
        sample_label = labels[idx]
        
        return {
            'input': sample_data,
            'target': sample_label,
            'equation': f"{sample_data[0].item()} + {sample_data[1].item()} = {sample_label.item()} (mod {self.modulo})"
        }

class ModularSubtractionDataset: #same exact thing as ModularAdditionDataset but with - instead of +, TEST IN WULVER 
    """Dataset generator for modular addition problems."""
    
    def __init__(self, data_cfg):
        self.modulo = data_cfg['modulo'] # the numbed to divide by
        self.train_frac = data_cfg['train_frac'] # how much data to use for training
        self.data_seed = data_cfg['data_seed'] # for repeatable results 
        self.operation = data_cfg['operation']
        
        # Generate the complete dataset
        self.dataset, self.labels = self._generate_dataset()
        
        # Split into train/test
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._split_dataset()
    
    def _generate_dataset(self):
        """Generate all possible modular substraction pairs."""
        # Create all combinations of a - b = c (mod MODULO)
        a_vector = einops.repeat(torch.arange(self.modulo), "i -> (i j)", j=self.modulo) # vectors a and b cover all possible pairs. ex: [0,0],[0,1],[1,2] ...
        b_vector = einops.repeat(torch.arange(self.modulo), "j -> (i j)", i=self.modulo)
        equals_vector = einops.repeat(torch.tensor(self.modulo), " -> (i j)", i=self.modulo, j=self.modulo)
        
        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)
        labels = (dataset[:, 0] - dataset[:, 1]) % self.modulo
        
        return dataset, labels
    
    def _split_dataset(self):
        """Split dataset into train and test sets."""
        torch.manual_seed(self.data_seed)
        
        indices = torch.randperm(self.modulo * self.modulo)
        cutoff = int(self.modulo * self.modulo * self.train_frac)
        
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
        
        train_data = self.dataset[train_indices]
        train_labels = self.labels[train_indices]
        test_data = self.dataset[test_indices]
        test_labels = self.labels[test_indices]
        
        return train_data, train_labels, test_data, test_labels
    
    def to_device(self, device):
        """Move all data to specified device."""
        self.dataset = self.dataset.to(device)
        self.labels = self.labels.to(device)
        self.train_data = self.train_data.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_data = self.test_data.to(device)
        self.test_labels = self.test_labels.to(device)
        return self
    
    def get_train_data(self):
        """Get training data and labels."""
        return self.train_data, self.train_labels
    
    def get_test_data(self):
        """Get test data and labels."""
        return self.test_data, self.test_labels
    
    def get_full_data(self):
        """Get complete dataset and labels."""
        return self.dataset, self.labels
    
    def get_data_info(self):
        """Get information about the dataset."""
        return {
            'modulo': self.modulo,
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_frac': self.train_frac,
            'vocab_size': self.modulo + 1  # +1 for the equals token
        }
    
    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the dataset for inspection."""
        if subset == 'train':
            data, labels = self.train_data, self.train_labels
        elif subset == 'test':
            data, labels = self.test_data, self.test_labels
        else:
            data, labels = self.dataset, self.labels
        
        if idx is None:
            idx = 0
        
        sample_data = data[idx]
        sample_label = labels[idx]
        
        return {
            'input': sample_data,
            'target': sample_label,
            'equation': f"{sample_data[0].item()} - {sample_data[1].item()} = {sample_label.item()} (mod {self.modulo})"
        }

class ModularMultiplicationDataset:
    """Dataset generator for modular multiplication problems."""
    
    def __init__(self, data_cfg):
        self.modulo = data_cfg['modulo'] # the number to divide by
        self.train_frac = data_cfg['train_frac'] # how much data to use for training
        self.data_seed = data_cfg['data_seed'] # for repeatable results
        self.operation = data_cfg['operation']
        
        # Generate the complete dataset
        self.dataset, self.labels = self._generate_dataset()
        
        # Split into train/test
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._split_dataset()

    def _generate_dataset(self):
        """Generate all possible modular multiplication pairs."""
        # Create all combinations of a * b = c (mod MODULO)
        a_vector = einops.repeat(torch.arange(1, self.modulo), "i -> (i j)", j=self.modulo-1) # vectors a and b cover all possible pairs. ex: [0,0],[0,1],[1,2] ...
        b_vector = einops.repeat(torch.arange(1, self.modulo), "j -> (i j)", i=self.modulo-1)
        equals_vector = einops.repeat(torch.tensor(self.modulo), " -> (i j)", i=self.modulo-1, j=self.modulo-1)
        
        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)
        labels = (dataset[:, 0] * dataset[:, 1]) % self.modulo

        dataset -= 1 #when training ml you want it to start at 0
        labels -= 1
        
        return dataset, labels

    def _split_dataset(self):
        """Split dataset into train and test sets."""
        torch.manual_seed(self.data_seed)
        
        indices = torch.randperm((self.modulo - 1) * (self.modulo-1))
        cutoff = int(len(indices) * self.train_frac)
        
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
        
        train_data = self.dataset[train_indices]
        train_labels = self.labels[train_indices]
        test_data = self.dataset[test_indices]
        test_labels = self.labels[test_indices]
        
        return train_data, train_labels, test_data, test_labels

    def to_device(self, device):
        """Move all data to specified device."""
        self.dataset = self.dataset.to(device)
        self.labels = self.labels.to(device)
        self.train_data = self.train_data.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_data = self.test_data.to(device)
        self.test_labels = self.test_labels.to(device)
        return self

    def get_train_data(self):
        """Get training data and labels."""
        return self.train_data, self.train_labels

    def get_test_data(self):
        """Get test data and labels."""
        return self.test_data, self.test_labels

    def get_full_data(self):
        """Get complete dataset and labels."""
        return self.dataset, self.labels

    def get_data_info(self):
        """Get information about the dataset."""
        return {
            'modulo': self.modulo,
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_frac': self.train_frac,
            'vocab_size': self.modulo  # 1,...,modulo-1 for the numbers and one more equals token
        }

    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the dataset for inspection."""
        if subset == 'train':
            data, labels = self.train_data, self.train_labels
        elif subset == 'test':
            data, labels = self.test_data, self.test_labels
        else:
            data, labels = self.dataset, self.labels
        
        if idx is None:
            idx = 0
        
        sample_data = data[idx]
        sample_label = labels[idx]
        
        return {
            'input': sample_data,
            'target': sample_label,
            'equation': f"{sample_data[0].item() + 1} * {sample_data[1].item() + 1} = {sample_label.item() + 1} (mod {self.modulo})"
        }

class ModularDivisionDataset:
    """Dataset generator for modular multiplication problems."""
    #same code
    def __init__(self, data_cfg):
        self.modulo = data_cfg['modulo'] # the number to divide by
        self.train_frac = data_cfg['train_frac'] # how much data to use for training
        self.data_seed = data_cfg['data_seed'] # for repeatable results
        self.operation = data_cfg['operation']
        
        # Generate the complete dataset
        self.dataset, self.labels = self._generate_dataset()
        
        # Split into train/test
        self.train_data, self.train_labels, self.test_data, self.test_labels = self._split_dataset()
    
    def _gcd(self, a, b):
        """Calculate Greatest Common Divisor using Euclidean algorithm."""
        while b:
            a, b = b, a % b
        return a

    def _mod_inverse(self, a, m):
        """Calculate modular multiplicative inverse using Extended Euclidean Algorithm."""
        if self._gcd(a, m) != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")
        
        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, y = extended_gcd(a, m)
        # Make sure the result is positive
        return (x % m + m) % m
    
    # a lot of change needed
    def _generate_dataset(self):
        """Generate all possible modular division pairs: a ÷ b ≡ c (mod modulo)."""
        
        # Step 1: Find all numbers that have modular multiplicative inverses
        # A number b has an inverse mod m if and only if gcd(b, m) = 1
        valid_divisors = []
        for b in range(1, self.modulo):
            if self._gcd(b, self.modulo) == 1:
                valid_divisors.append(b)
        
        # Step 2: Generate all valid combinations
        dataset_list = []
        labels_list = []
        
        for a in range(1, self.modulo):  # dividend from 1 to modulo-1
            for b in valid_divisors:      # divisor must have inverse
                # Calculate c = a ÷ b ≡ a × b⁻¹ (mod modulo)
                b_inverse = self._mod_inverse(b, self.modulo)
                c = (a * b_inverse) % self.modulo
                
                # Handle the case where result is 0 (should map to modulo)
                if c == 0:
                    c = self.modulo
                
                # Create the input format [a, b, equals_token]
                equals_token = self.modulo
                dataset_list.append([a, b, equals_token])
                labels_list.append(c)

        # Convert to tensors
        dataset = torch.tensor(dataset_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        
        # Convert to 0-indexed (subtract 1 from everything)
        dataset -= 1
        labels -= 1
        
        return dataset, labels

    

    def _split_dataset(self):
        """Split dataset into train and test sets."""
        torch.manual_seed(self.data_seed)
        
        indices = torch.randperm((self.modulo - 1) * (self.modulo-1))
        cutoff = int(len(indices) * self.train_frac)
        
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]
        
        train_data = self.dataset[train_indices]
        train_labels = self.labels[train_indices]
        test_data = self.dataset[test_indices]
        test_labels = self.labels[test_indices]
        
        return train_data, train_labels, test_data, test_labels

    def to_device(self, device):
        """Move all data to specified device."""
        self.dataset = self.dataset.to(device)
        self.labels = self.labels.to(device)
        self.train_data = self.train_data.to(device)
        self.train_labels = self.train_labels.to(device)
        self.test_data = self.test_data.to(device)
        self.test_labels = self.test_labels.to(device)
        return self

    def get_train_data(self):
        """Get training data and labels."""
        return self.train_data, self.train_labels

    def get_test_data(self):
        """Get test data and labels."""
        return self.test_data, self.test_labels

    def get_full_data(self):
        """Get complete dataset and labels."""
        return self.dataset, self.labels

    def get_data_info(self):
        """Get information about the dataset."""
        return {
            'modulo': self.modulo,
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_frac': self.train_frac,
            'vocab_size': self.modulo  # 1,...,modulo-1 for the numbers and one more equals token
        }

    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the dataset for inspection."""
        if subset == 'train':
            data, labels = self.train_data, self.train_labels
        elif subset == 'test':
            data, labels = self.test_data, self.test_labels
        else:
            data, labels = self.dataset, self.labels
        
        if idx is None:
            idx = 0
        
        sample_data = data[idx]
        sample_label = labels[idx]
        
        return {
            'input': sample_data,
            'target': sample_label,
            'equation': f"{sample_data[0].item() + 1} ÷ {sample_data[1].item() + 1} = {sample_label.item() + 1} (mod {self.modulo})"
        }