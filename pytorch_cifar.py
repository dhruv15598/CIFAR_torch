import torch
import torch.optim as optim
import argparse
import ssl
import os
import matplotlib.pyplot as plt
from model import ConvNet
from utils import (setup_data_loaders, save_model, load_model, evaluate_model,
                   test_model_comprehensive, train_model, plot_results, visualize_model_features,
                   setup_training, visualize_misclassified)

# SSL certificate verification fix for macOS
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    # Replace the args = parse_arguments() with direct settings
    class Args:
        def __init__(self):
            self.mode = 'train'  # 'train', 'visualize', 'test', 'both'
            self.model_path = 'model_checkpoint_v2.pth'
            self.model_name = 'best_model_v2.pth'
            self.learning_rate = 0.001
            self.epochs = 10
            self.batch_size = 64

    args = Args()  # Use this instead of parse_arguments()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model and optimizer
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Variables to store training history
    start_epoch = 0
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # Load existing model if available
    if args.mode in ['visualize', 'test', 'both']:
        if os.path.exists(args.model_path):
            print(f"Loading existing model from {args.model_path}")
            start_epoch, train_losses, train_accuracies, test_accuracies = load_model(
                model, optimizer, args.model_path)
        else:
            print(f"No model found at {args.model_path}. Please train the model first.")
            return

    # Setup data loaders
    train_loader, test_loader = setup_data_loaders(args.batch_size)

    # Training mode
    if args.mode in ['train']:
        print("\nStarting training...")
        new_losses, new_train_acc, new_test_acc = train_model(
            model, device, train_loader, test_loader,
            args.epochs, optimizer, args.model_name, start_epoch)

        # Update metrics
        train_losses.extend(new_losses)
        train_accuracies.extend(new_train_acc)
        test_accuracies.extend(new_test_acc)

        # Plot training results
        plot_results(train_losses, train_accuracies, test_accuracies)

    # Testing mode
    if args.mode in ['test', 'both']:
        print("\nRunning comprehensive model evaluation...")
        # Overall model performance
        test_model_comprehensive(model, device, test_loader)

        # Show misclassified examples
        print("\nShowing misclassified examples...")
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        visualize_misclassified(model, device, test_loader, class_names)

        # Calculate and print test accuracy
        test_acc = evaluate_model(model, device, test_loader)
        print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # Visualization mode
    if args.mode in ['visualize', 'both']:
        print("\nVisualizing model features...")
        visualize_model_features(model, device)


if __name__ == '__main__':
    main()