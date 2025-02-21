import torch.cuda

#Configuration file for Age Estimation model
config = {
    # Dimensions for input images
    'img_width': 128, # Width of the input images
    'img_height': 128, # Height of the input images
    'img_size': 128, # Size of the input images

    # Normalization parameters for the images
    'mean': [0.485, 0.456, 0.406],  # Mean values for normalization 
    'std': [0.229, 0.224, 0.225], # Standard deviation values for normalization 
    
    # Model configuration
    'model_name': 'resnet',  # Name of the model to be used

    # Device configuration
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # File paths for input and output images(for Testing)
    'image_path_test': '../Tested_img/test_3.png',
    'output_path_test': '../Tested_img/test_3output.png',
    'leaky_relu': False, # Flag to indicate if Leaky ReLU should be used as an activation function
    # Training parameters
    'epochs': 50,  # Number of training epochs
    'batch_size': 128,  # Batch size for training
    'eval_batch_size': 256,  # Batch size for evaluation
    'seed': 42,  # Random seed value

    # Learning rate and weight decay for the optimizer
    'lr': 0.0001,  # Learning rate for the optimizer
    'wd': 0.001  # Weight decay (L2 regularization) for the optimizer
}
