import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class EthnicityClassifier(nn.Module):
  def __init__(self,num_classes=5):
    super(EthnicityClassifier, self).__init__()
    self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

  def forward(self, x):
        return self.model(x)
