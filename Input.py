# Imports
import torch
from zipfile import ZipFile
from torchvision import transforms, datasets

# Number of input data in a single batch
BATCH_SIZE = 100

# dataset root directory
data_dir = 'mask_dataset'

# directory to training dataset
train_dir = data_dir + '/train'

# directory to validation dataset
valid_dir = data_dir + '/valid'

# directory to testing dataset
test_dir = data_dir + '/test'


def unzip():
    """Extracting data from zip file"""
    with ZipFile('/content/drive/My Drive/mask_dataset.zip', 'r') as zipped_file:
        zipped_file.extractall(data_dir)


# transforms to be applied to training and validation dataset
train_transforms = transforms.Compose([transforms.RandomPerspective(),
                                       transforms.Resize((100, 100)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.4437, 0.3848, 0.3613], [0.2972, 0.2702, 0.2581])
                                       ])

# transforms to be applied to testing dataset
test_transforms = transforms.Compose([transforms.Resize((100, 100)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4437, 0.3848, 0.3613], [0.2972, 0.2702, 0.2581])
                                      ])

# Loading training dataset
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

# Loading validation dataset
valid_dataset = datasets.ImageFolder(valid_dir, transform=train_transforms)

# Loading testing dataset
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Dataloader for training set
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Dataloader for validation set
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# Dataloader for testing set
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
