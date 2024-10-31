import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from tqdm import tqdm
import os
from model import ConvNet
from utils import (setup_data_loaders, save_model, load_model, test_model_comprehensive)


class Trainer:
    def __init__(self, model, device, train_loader, test_loader, optimizer, criterion=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # Metrics storage
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.best_accuracy = 0.0
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    def train_epoch(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Use tqdm for progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                    desc=f'Epoch {epoch + 1}/{num_epochs}')

        for i, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics calculation
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total

        # Calculate precision, recall, and F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')

        return {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100 * correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, num_epochs, start_epoch=0):
        for epoch in range(start_epoch, num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(epoch, num_epochs)

            # Evaluate on test set
            test_metrics = self.evaluate()

            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.test_accuracies.append(test_metrics['accuracy'])
            self.precisions.append(test_metrics['precision'])
            self.recalls.append(test_metrics['recall'])
            self.f1_scores.append(test_metrics['f1'])

            # Adjust learning rate
            self.scheduler.step(train_metrics['loss'])

            # Print metrics
            print(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Train Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"Test F1 Score: {test_metrics['f1']:.4f}")

            # Save checkpoint
            save_model(self.model, self.optimizer, epoch + 1,
                       self.train_losses, self.train_accuracies, self.test_accuracies)

            # Save best model
            if test_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = test_metrics['accuracy']
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"New best model saved! Accuracy: {self.best_accuracy:.2f}%")

            # Plot current progress
            if (epoch + 1) % 5 == 0:  # Plot every 5 epochs
                self.plot_training_progress()

        return self.train_losses, self.train_accuracies, self.test_accuracies

    def plot_training_progress(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plot
        axes[0, 0].plot(self.train_losses)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')

        # Accuracy plot
        axes[0, 1].plot(self.train_accuracies, label='Train')
        axes[0, 1].plot(self.test_accuracies, label='Test')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()

        # F1 Score plot
        axes[1, 0].plot(self.f1_scores)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')

        # Precision-Recall plot
        axes[1, 1].plot(self.precisions, label='Precision')
        axes[1, 1].plot(self.recalls, label='Recall')
        axes[1, 1].set_title('Precision and Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    # Initialize model and optimizer
    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Setup data
    train_loader, test_loader = setup_data_loaders(batch_size)

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer
    )

    # Load existing model or train from scratch
    model_path = 'model_checkpoint.pth'
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        start_epoch, losses, train_acc, test_acc = load_model(model, optimizer, model_path)
        trainer.train_losses = losses
        trainer.train_accuracies = train_acc
        trainer.test_accuracies = test_acc
    else:
        start_epoch = 0

    # Train model
    print("\nStarting training...")
    trainer.train(num_epochs, start_epoch)

    # Final evaluation
    print("\nRunning final comprehensive evaluation...")
    test_model_comprehensive(model, device, test_loader)


if __name__ == '__main__':
    main()