import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class UNet(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()
        pass

    def forward(self, x):
        pass

def load_model(checkpoint_path, device):
    model = UNet(num_classes=1)  
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])  
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, default="model.pth")
    parser.add_argument('--output_path', type=str, default="segmented_output.png")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.checkpoint_path, device)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(args.image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output_mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)  

    output_image = Image.fromarray(output_mask * 255)  
    output_image.save(args.output_path)
    print(f"Segmented output saved to {args.output_path}")
