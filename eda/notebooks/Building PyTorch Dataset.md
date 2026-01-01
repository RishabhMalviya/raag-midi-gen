pytorch dataset multiple tensors in each data and label


In PyTorch, you can return multiple tensors for both data (inputs) and labels (targets) by customizing the `__getitem__` method of a `torch.utils.data.Dataset` subclass. 

## 1. Using a Custom Dataset Class
A custom dataset allows you to return any structure (tuples or dictionaries) containing multiple tensors. The DataLoader will automatically collate these into batches with the same structure. 
```python
import torch
from torch.utils.data import Dataset, DataLoader

class MultiTensorDataset(Dataset):
    def __init__(self, data1, data2, label1, label2):
        self.data1 = data1
        self.data2 = data2
        self.label1 = label1
        self.label2 = label2

    def __len__(self):
        return len(self.label1)

    def __getitem__(self, idx):
        # Return multiple tensors as a tuple or dictionary
        sample = {
            'inputs': (self.data1[idx], self.data2[idx]),
            'labels': (self.label1[idx], self.label2[idx])
        }
        return sample

# Usage
dataset = MultiTensorDataset(d1, d2, l1, l2)
loader = DataLoader(dataset, batch_size=32)

for batch in loader:
    input1, input2 = batch['inputs']
    target1, target2 = batch['labels']
    # Pass to model
```

## 2. Using TensorDataset (For simple cases)
If all your data and label parts are already pre-loaded into memory as tensors of equal length, you can use the built-in TensorDataset. It accepts an arbitrary number of tensors and returns them as a tuple in each iteration. 
```python
from torch.utils.data import TensorDataset

# Pass all input and label tensors to the constructor
dataset = TensorDataset(data_tensor1, data_tensor2, label_tensor1, label_tensor2)
loader = DataLoader(dataset, batch_size=32)

for d1, d2, l1, l2 in loader:
    # d1 and d2 are your inputs; l1 and l2 are your labels
    pass
```

## 3. Handling Variable Lengths
If your multiple tensors have different lengths (e.g., varying number of tags per image), you cannot stack them into a standard batch directly. In this case: 
- **One-Hot Encoding**: Convert labels to a fixed-length vector representing all possible classes.
- **Padding**: Use torch.nn.utils.rnn.pad_sequence or a custom collate_fn to pad sequences to the same length within a batch. 
#### Key Implementation Tips
- **Collation**: By default, the DataLoader stacks tensors into a new dimension (the batch dimension).
- **Dictionaries**: Returning a dictionary in `__getitem__` is often cleaner for complex models with many inputs and outputs.
- **Performance**: For very large datasets, avoid loading all tensors into memory at once; instead, load file paths in `__init__` and read the actual data in `__getitem__`. 

Would you like an example of a custom collate function to handle tensors that have different shapes or lengths?