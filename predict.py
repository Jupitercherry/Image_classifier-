# Python
import argparse
import torch
from torchvision import transforms  
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import json

def process_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def predict(image_path, arch,model_checkpoint, topk, category_names,gpu):
    device = torch.device("cuda" if (torch.cuda.is_available() & gpu == True) else "cpu")
    print (f"Using device: {device}")

    
    # Load and preprocess the image
    image = process_image(image_path).unsqueeze(0)
    image = image.to(device)
    # Load the model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    if arch == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1) )
    elif arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)     )
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1) )
    else:
        raise ValueError("Unsupported architecture. Choose from 'alexnet', 'vgg16', or 'resnet18'.")
    

    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = optim.SGD(model.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_p, top_class = probabilities.topk(topk, dim=1)

    # Convert indices to class labels
    with open(category_names, 'r') as f:
         cat_to_name = json.load(f)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[int(i)] for i in top_class[0]]
    class_labels = [idx_to_class[c] if isinstance(c, int) else c for c in top_classes]
    class_names = [cat_to_name[str(label)] for label in class_labels]

    return top_p[0].cpu().numpy(), top_classes, class_names


def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('arch', type=str, default='alexnet', help='Model architecture (default: alexnet)')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default=None, help='Path to category to name mapping JSON file')
    parser.add_argument('--gpu', action='store_true',default=False , help='Use GPU for inference')
    args = parser.parse_args()
    top_p, top_classes,class_names = predict(args.image_path, ardgs.arch,args.checkpoint, args.top_k,args.category_names, args.gpu)
    print(f'Top {args.top_k} classes: {top_classes}')
    print(f'Top {args.top_k} probabilities: {top_p}')
    print(f'Top {args.top_k} class names: {class_names}')

if __name__ == "__main__":
    main()