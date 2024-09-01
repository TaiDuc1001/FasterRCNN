import torch
from models.fasterrcnn import TwoStageDetector
from models.utils import project_bboxes, display_bbox, display_img
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define image dimensions
    channels = 3
    height = 224
    width = 224
    image = Image.open("pedestrian-accident-injured.jpg").resize((width, height))
    image = np.array(image)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
    print("Image shape:", image.shape)
    
    # Create a fake image
    # fake_image = create_fake_image(batch_size, channels, height, width)
    
    # fake_gt_boxes = torch.tensor([
    #     [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
    #     [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]
    # ])
    # fake_gt_classes = torch.tensor([[0, 1], [1, 0]])
    
    # Initialize the TwoStageDetector
    img_size = (height, width)
    out_size = (7, 7)
    hs, ws = height // out_size[0], width // out_size[1]
    out_channels = 2048
    n_classes = 91  
    roi_size = (2, 2)
    detector = TwoStageDetector(
        img_size=img_size, 
        out_size=out_size, 
        out_channels=out_channels, 
        n_classes=n_classes, 
        roi_size=roi_size
    )

    detector.eval()
    proposals_final, conf_scores_final, \
    classes_final, probabs_final, embeddings_final = detector.inference(
        images=image, 
        nms_thresh=0.05, 
        conf_thresh=0.9
        )
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig, ax = display_img(image, fig, ax)
    for i in range(len(proposals_final)):
        prop_proj = project_bboxes(proposals_final[i], ws, hs, mode="a2p")
        # print("Proposals projected:", prop_proj)
        fig, _ = display_bbox(prop_proj, fig, ax, color="red")
    plt.show()



if __name__ == '__main__':
    main()