import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import os
from Age_config import config

# read csv that we created in `create_csv.py` file
df = pd.read_csv('./csv_dataset_AgeDetect/utkface_dataset.csv')

df_train, df_test = train_test_split(df, train_size=0.8, random_state=config['seed'])
df_train, df_valid = train_test_split(df_train, train_size=0.85, random_state=config['seed'])

# Save the training, validation, and test sets in separate CSV files.
directory = './csv_dataset_AgeDetect'

if not os.path.exists(directory):
    os.makedirs(directory)

df_train.to_csv(os.path.join(directory, 'train_set.csv'), index=False)
df_valid.to_csv(os.path.join(directory, 'valid_set.csv'), index=False)
df_test.to_csv(os.path.join(directory, 'test_set.csv'), index=False)

print('All CSV files created successfully.')

# Define transformations
transform_train = T.Compose([T.Resize((config['img_width'], config['img_height'])),
                             T.RandomHorizontalFlip(),
                             T.RandomRotation(degrees=15),
                             T.ColorJitter(brightness=(0.5, 1.5), contrast=1, saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
                             T.ToTensor(),
                             T.Normalize(mean=config['mean'], std=config['std'])
                             ])

transform_test = T.Compose([T.Resize((config['img_width'], config['img_height'])),
                            T.ToTensor(),
                            T.Normalize(mean=config['mean'], std=config['std'])
                            ])


# Custom dataset: A custom dataset class for UTKFace.
class UTKDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one_row = self.data.iloc[idx].values
        img_dir = os.path.join(self.root_dir, one_row[0])

        image = Image.open(img_dir).convert('RGB')
        image = self.transform(image)
        age = torch.tensor([one_row[1]], dtype=torch.float32)

        gender = one_row[2]
        ethnicity = one_row[3]

        return image, age, gender, ethnicity


# Utilize the UTKDataset class to instantiate dataset objects for the training, validation, and test sets.
root_dir = '../Datasets/utkcropped'
csv_file_train = './csv_dataset_AgeDetect/train_set.csv'
csv_file_valid = './csv_dataset_AgeDetect/valid_set.csv'
csv_file_test = './csv_dataset_AgeDetect/test_set.csv'

# Define dataloader: Write dataloaders for the training, validation, and test sets.
train_set = UTKDataset(root_dir, csv_file_train, transform_train)
valid_set = UTKDataset(root_dir, csv_file_valid, transform_test)
test_set = UTKDataset(root_dir, csv_file_test, transform_test)

train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=config['eval_batch_size'])
test_loader = DataLoader(test_set, batch_size=config['eval_batch_size'])

