import os
import torch
import torchvision
import json
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import v2 as T
from torchsummary import summary
from torch.nn import GaussianNLLLoss



model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
# summary(model)
model.train()

# for param in model.parameters():
#     hidden = param.shape
# print(model.classifier)
# print(model.classifier[0])
head = torch.nn.Linear(model.classifier[3].out_features, 2)

def forward(input):
    x = model(input)
    return head(x)

optimizer = torch.optim.AdamW(params = None, lr = 1e-3)
outputs = forward(input)
loss = GaussianNLLLoss(outputs, ground_truth)
loss.backward()
optimizer.step()

