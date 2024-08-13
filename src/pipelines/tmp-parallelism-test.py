import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import copy


# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)
    def __getitem__(self, index):
        return {
            'input': copy.copy(self.data[index]),
            'input_2': copy.copy(self.data[index]),
            'index_feature': torch.tensor(list(range(len(self.data[index])))),
            'extra_feature': [1, 2, 3, 4]
        }
    def __len__(self):
        return self.len

rand_loader = DataLoader(
    dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True
)

class Model(nn.Module):
    # Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, input, input_2, index_feature, extra_feature):
        output = self.fc(input)
        print(len(extra_feature))
        print(
            "\tIn Model: input size", input.size(),
            "\tIn Model: input_2 size", input_2.size(),
            "\tIn Model: index_feature size", index_feature.size(),
            "output size", output.size()
              )
        return output


model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)
model.to(device)

for data in rand_loader:
    data['input'] = data['input'].to(device)
    output = model(**data)
    print("Outside: input size", input.size(),
          "output_size", output.size())