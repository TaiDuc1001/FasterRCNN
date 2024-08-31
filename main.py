import torch
from models.fasterrcnn import TwoStageDetector

def create_fake_image(batch_size, channels, height, width):
    # Create a random tensor to simulate an image
    return torch.randn(batch_size, channels, height, width)

def main():
    # Define image dimensions
    batch_size = 1
    channels = 3
    height = 224
    width = 224
    
    # Create a fake image
    fake_image = create_fake_image(batch_size, channels, height, width)
    
    # Initialize the TwoStageDetector
    img_size = (height, width)
    out_size = (7, 7)
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
    # Perform inference
    proposals_final, conf_scores_final, classes_final = detector.inference(
        fake_image, 
        nms_thresh=0.05, 
        conf_thresh=0.1
        )

    # Print the results
    print("Proposals:", proposals_final)
    print("Confidence Scores:", conf_scores_final)
    print("Classes:", classes_final)

if __name__ == '__main__':
    main()