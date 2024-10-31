import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from torch.nn import functional as F


def train_model(model, device, train_loader, test_loader, num_epochs, optimizer, start_epoch=0):
    criterion = nn.CrossEntropyLoss()

    # Create the OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Step the scheduler
            scheduler.step()  # Move scheduler.step() here for OneCycleLR

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Test accuracy
        test_accuracy = evaluate_model(model, device, test_loader)
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
              f'Train Accuracy: {epoch_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

        # Save checkpoint
        save_model(model, optimizer, epoch + 1, train_losses, train_accuracies, test_accuracies)

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), 'best_model_v1.pth')

    return train_losses, train_accuracies, test_accuracies


def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def plot_results(train_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, 'g-', label='Train')
    plt.plot(test_accuracies, 'r-', label='Test')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


class FeatureExtractor:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = {name: None for name in layer_names}

        for name, module in self.model.named_modules():
            if name in layer_names:
                module.register_forward_hook(self.get_feature_hook(name))

    def get_feature_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output.detach()

        return hook

    def get_features(self, x):
        _ = self.model(x)
        return self.features


def setup_data_loaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Increased rotation
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added hue
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Add random crop
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform_train, download=True)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def save_model(model, optimizer, epoch, train_losses, train_accuracies, test_accuracies,
               filename='model_checkpoint_v1.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


def load_model(model, optimizer, filename='model_checkpoint.pth'):
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        print(f"Model loaded from {filename}")
        return epoch, train_losses, train_accuracies, test_accuracies
    else:
        print(f"No checkpoint found at {filename}")
        return 0, [], [], []


def evaluate_model(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def visualize_filters(model, layer_name, num_filters=16):
    conv_layer = None
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Conv2d):
            conv_layer = module
            break

    if conv_layer is None:
        print(f"Layer {layer_name} not found or is not a convolutional layer")
        return

    weights = conv_layer.weight.data.cpu()
    weights = weights - weights.min()
    weights = weights / weights.max()

    plt.figure(figsize=(20, 10))
    for i in range(min(num_filters, weights.shape[0])):
        plt.subplot(2, num_filters // 2, i + 1)
        if weights.shape[1] == 3:
            img = weights[i].permute(1, 2, 0)
        else:
            img = weights[i].mean(0)
        plt.imshow(img, cmap='viridis' if weights.shape[1] != 3 else None)
        plt.axis('off')
        plt.title(f'Filter {i + 1}')
    plt.tight_layout()
    plt.show()


def get_sample_images(num_images=5):
    """Get sample images from CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        transform=transform, download=True
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_images, shuffle=True
    )

    images, labels = next(iter(loader))
    return images, labels


def visualize_feature_maps(model, image, layer_names):
    model.eval()
    feature_extractor = FeatureExtractor(model, layer_names)

    with torch.no_grad():
        features = feature_extractor.get_features(image.unsqueeze(0))

    for layer_name in layer_names:
        feature_maps = features[layer_name][0]
        num_maps = min(16, feature_maps.shape[0])
        grid_size = int(np.ceil(np.sqrt(num_maps)))

        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Feature Maps for {layer_name}')

        for i in range(num_maps):
            plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(feature_maps[i].cpu(), cmap='viridis')
            plt.axis('off')
            plt.title(f'Map {i + 1}')
        plt.tight_layout()
        plt.show()


def visualize_model_features(model, device):
    """Simplified visualization for the 2-block CNN"""
    # 1. Visualize convolutional filters
    print("Visualizing filters from different layers...")
    conv_layers = [
        'conv_layers.0',  # First conv layer
        'conv_layers.4'  # Second conv layer
    ]

    for layer_name in conv_layers:
        print(f"\nFilters in {layer_name}:")
        visualize_filters(model, layer_name)

    # 2. Visualize feature maps for sample images
    print("\nVisualizing feature maps for sample images...")
    images, labels = get_sample_images(1)
    images = images.to(device)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Show original image
    plt.figure(figsize=(5, 5))
    img = images[0].cpu().numpy().transpose(1, 2, 0)
    img = img * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f'Input Image: {class_names[labels[0]]}')
    plt.axis('off')
    plt.show()

    # Visualize feature maps for both conv layers
    layer_names = [
        'conv_layers.0',  # First conv layer
        'conv_layers.4'  # Second conv layer
    ]
    visualize_feature_maps(model, images[0], layer_names)


def test_model_comprehensive(model, device, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion Matrix
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=class_names, y=per_class_acc)
    plt.title('Per-class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Show misclassified examples
    visualize_misclassified(model, device, test_loader, class_names)
    analyze_model_confidence(model, device, test_loader)


def visualize_misclassified(model, device, test_loader, class_names, num_images=10):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            mask = (predicted != labels)
            misclassified_images.extend(images[mask].cpu())
            misclassified_labels.extend(labels[mask].cpu())
            predictions.extend(predicted[mask].cpu())

            if len(misclassified_images) >= num_images:
                break

    plt.figure(figsize=(20, 4))
    for i in range(min(num_images, len(misclassified_images))):
        plt.subplot(2, 5, i + 1)
        img = misclassified_images[i].numpy().transpose(1, 2, 0)
        img = img * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f'True: {class_names[misclassified_labels[i]]}\n' +
                  f'Pred: {class_names[predictions[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def setup_training():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    # Image preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transform_train, download=True)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transform_test, download=True)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return device, num_epochs, train_loader, test_loader


def analyze_model_confidence(model, device, test_loader):
    model.eval()
    confidences = {i: [] for i in range(10)}
    correct_confidences = []
    incorrect_confidences = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)

            for i, (prob, pred, label) in enumerate(zip(max_probs, predicted, labels)):
                confidences[label.item()].append(prob.item())
                if pred == label:
                    correct_confidences.append(prob.item())
                else:
                    incorrect_confidences.append(prob.item())

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=50, alpha=0.5, label='Correct', density=True)
    plt.hist(incorrect_confidences, bins=50, alpha=0.5, label='Incorrect', density=True)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.legend()

    plt.subplot(1, 2, 2)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    box_data = [confidences[i] for i in range(10)]
    plt.boxplot(box_data, labels=class_names)
    plt.title('Confidence Distribution per Class')
    plt.xticks(rotation=45)
    plt.ylabel('Confidence')

    plt.tight_layout()
    plt.show()
