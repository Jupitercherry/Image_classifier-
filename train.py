import argparse
import torch 
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision import models
from torchvision.models import AlexNet_Weights, VGG16_Weights, ResNet18_Weights

def train(dataset, arch, learning_rate, epochs, hidden_units, gpu,save_dir):


    device = torch.device("cuda" if (torch.cuda.is_available() & gpu) else "cpu")
    print(f"Using device: {device}")

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset
    train_data = datasets.ImageFolder(root=dataset + '/train', transform=data_transforms)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True) 
    valid_data = datasets.ImageFolder(root=dataset + '/valid' ,transform=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False) 

    
    if arch== "alexnet":
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
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
            nn.LogSoftmax(dim=1)        
        )
    else:
        raise ValueError("Unsupported architecture. Choose from 'alexnet', 'vgg16', or 'resnet18'.")        
    model.to(device)
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)     
    # Training loop
    train_losses = []
    val_losses = []
    for e in range(epochs):
         running_loss = 0
         model.train()
         for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
         train_losses.append(running_loss / len(train_loader))

         print("Epoch: {}/{}.. ".format(e + 1, epochs),
          "Training Loss: {:.3f}.. ".format(train_losses[-1]))

    # Validation phase
         val_loss = 0
         accuracy = 0
         model.eval()
         with torch.no_grad():
            for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels).item()
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.float()).item()
            val_losses.append(val_loss / len(valid_loader))
            print("Validation Loss: {:.3f}.. ".format(val_losses[-1]),
            "Validation Accuracy: {:.3f}".format(accuracy / len(valid_loader)))
        
    # Save the model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'arch': arch,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'hidden_units': hidden_units    

    }
    torch.save(checkpoint, save_dir)
    print(f"Model checkpoint saved as '{save_dir}'.") 
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model with the dataset')
    parser.add_argument('dataset', type=str, help='Path to the training data directory')
    parser.add_argument('--arch', type=str, default='vgg16',help='vgg16, resnet18, or alexnet')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train the model')
    parser.add_argument('--hidden_units', type=int, default=512, help='number of hidden units in the classifier')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Set directory to save checkpoints')

    args = parser.parse_args()

    train(args.dataset, args.arch, args.learning_rate, args.epochs, args.hidden_units, args.gpu,args.save_dir)