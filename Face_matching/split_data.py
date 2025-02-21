import os
import random
import shutil

def split_lfw_folders(source_dir, train_dir, test_dir, test_ratio=0.2):
    """
    Splits LFW dataset by moving person folders randomly to train and test directories.
    
    Args:
        source_dir: Path to original LFW dataset directory
        train_dir: Path where training folders will be moved
        test_dir: Path where test folders will be moved
        test_ratio: Proportion of folders to use for testing (default 0.2)
    """
    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of all person folders
    person_folders = [f for f in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, f))]
    
    # Randomly shuffle the folders
    random.shuffle(person_folders)
    
    # Calculate split point
    split_idx = int(len(person_folders) * test_ratio)
    
    # Split folders into train and test sets
    test_folders = person_folders[:split_idx]
    train_folders = person_folders[split_idx:]
    
    # Move folders to their respective directories
    for folder in train_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(train_dir, folder)
        shutil.copytree(src, dst)
        print(f"Moved {folder} to train set")
        
    for folder in test_folders:
        src = os.path.join(source_dir, folder)
        dst = os.path.join(test_dir, folder)
        shutil.copytree(src, dst)
        print(f"Moved {folder} to test set")
    
    print(f"\nSplit completed:")
    print(f"Train set: {len(train_folders)} people")
    print(f"Test set: {len(test_folders)} people")


source_dir = "../Datasets/lfw-deepfunneled"  # Original LFW dataset path
train_dir = "./lfw-deepfunneled_splitted/train"         # Where train folders will go
test_dir = "./lfw-deepfunneled_splitted/test"           # Where test folders will go

split_lfw_folders(source_dir, train_dir, test_dir, test_ratio=0.2)