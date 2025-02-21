from torch import nn
from torchvision.models import resnet, efficientnet_b0
import timm
from Age_Estimator.Age_config import config

# Custom Model
class AgeEstimationModel(nn.Module):

    def __init__(self, input_dim, output_nodes, model_name, pretrain_weights):
        super(AgeEstimationModel, self).__init__()

        self.input_dim = input_dim
        self.output_nodes = output_nodes
        self.pretrain_weights = pretrain_weights

        if model_name == 'resnet':
            self.model = resnet.resnet50(weights=pretrain_weights)
            self.model.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                          nn.Linear(in_features=2048, out_features=256, bias=True),
                                          nn.Linear(in_features=256, out_features=self.output_nodes, bias=True))


        elif model_name == 'vit':
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', img_size=config['img_size'], pretrained=pretrain_weights)
            
            # num_features = model.blocks[11].mlp.fc2.out_features
            num_features = 384
            self.model.head = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                            nn.Linear(num_features, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, self.output_nodes))

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def forward(self, x):
        x = self.model(x)
        return x

