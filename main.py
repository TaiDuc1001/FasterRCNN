import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.fasterrcnn import RegionProposalNetwork
from torchvision import transforms
from PIL import Image

def main():
    model = RegionProposalNetwork()
    imgs = torch.randn(2, 3, 224, 224)  # Fake batch of 2 image tensors
    feats = torch.randn(2, 512, 14, 14)  # Fake batch of 2 feature map tensors
    targets = [
        {
            'bboxes': torch.tensor([[10, 10, 100, 100], [50, 50, 200, 200]]),
            'labels': torch.tensor([1, 2])
        },
        {
            'bboxes': torch.tensor([[30, 30, 150, 150]]),
            'labels': torch.tensor([1])
        }
    ]
    model.train()
    output = model(imgs, feats, targets)
    # print(output)

if __name__ == '__main__':
    main()