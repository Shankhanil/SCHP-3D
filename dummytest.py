from datasets.cloth_dataset import ClothDataset
import torch

dataset = ClothDataset( root="data", dataset="test" )
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(images)
# print('----------------')
print(labels)
# print(dataset.train_list)