import torch.cuda

#Configuration file for Age Estimation model
config = {
    # Device configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # File paths for input and output images(for Testing)
    'path_train_set': './lfw-deepfunneled_splitted/train',
    'path_test_set': './lfw-deepfunneled_splitted/test',
    # Training parameters
    'epochs': 50,  # Number of training epochs
    'batch_size':64,  # Batch size for training

    # Learning rate and weight decay for the optimizer
    'lr': 0.0001,  # Learning rate for the optimizer
    }
