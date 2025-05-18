import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from src.dataset import ChocolateDataset
# from src.model import SimpleCNN
# from src.model import SimpleCNN_Light
from src.model import DeepResidualCNN
# from src.model import SimpleResNet
# from src.model import DeepChocolateCounter
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
import pandas as pd
import os
import math
import argparse
from collections import Counter
import shutil

batch_size = 2

# Function to calculate the image-wise average F1-score
def calculate_f1_score(y_true, y_pred):
    # y_pred = np.argmax(y_pred, axis=-1) # Used for thge new architecture
    N, C = y_true.shape
    f1_scores = []
    y_pred = np.argmax(y_pred, axis=-1)  # Convert predictions to class labels
    # print(f"y_pred: {y_pred}", flush=True)

    for i in range(N):
        TP_i = sum(min(y_true[i, j], y_pred[i, j]) for j in range(C))
        FPN_i = sum(abs(y_true[i, j] - y_pred[i, j]) for j in range(C))
        f1_scores.append((2 * TP_i) / (2 * TP_i + FPN_i))
    return sum(f1_scores) / N

def main():
    # Add more transforms for data augmentation
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert to tensor
    #     # # transforms.RandomResizedCrop((224, 224)),  # Random cropping
    #     transforms.RandomHorizontalFlip(),  # Flip images horizontally
    #     transforms.RandomRotation(15),  # Rotate images randomly
    #     transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.1, hue=0.3),  # Adjust colors
    #     # transforms.RandomErasing(p=0.2),  # Randomly erase parts of the image
    # ])
    transform = transforms.Compose([
        # transforms.Resize((1024, 1024)),  # Resize images to 1024x1024
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ToTensor()
    ])
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    # ])
    
    train_csv = './data/train.csv'
    train_img_dir = './data/train'

    # Print when data has been loaded
    print("Loading training data...", flush=True)
    train_dataset = ChocolateDataset(train_csv, train_img_dir, transform = transform, preload=True)  # Preload images for faster training
    print("Training data loaded.", flush=True)

    # Re-incorporate 80/20 train-test split from the train set
    train_size = int(0.7 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    # Print the ids of the training dataset
    # Delete all folders and files in 'yeet' before saving new images
    if os.path.exists('training_samples'):
        shutil.rmtree('training_samples')
    os.makedirs('training_samples', exist_ok=True)

    for i in tqdm(range(len(train_dataset)), desc="Saving training images"):
        img, label = train_dataset[i]
        # img: Tensor [C, H, W], label: Tensor [13]
        # Convert tensor to PIL image for saving
        img_pil = transforms.ToPILImage()(img)
        img_save_path = os.path.join('training_samples', f'train_img_{i}.png')
        img_pil.save(img_save_path)
    print(f"Saved {len(train_dataset)} training images to 'training_samples' folder.", flush=True)
    # Save the images from the validation dataset
    if os.path.exists('validation_samples'):
        shutil.rmtree('validation_samples')
    os.makedirs('validation_samples', exist_ok=True)
    for i in tqdm(range(len(validation_dataset)), desc="Saving validation images"):
        img, label = validation_dataset[i]
        # img: Tensor [C, H, W], label: Tensor [13]
        # Convert tensor to PIL image for saving
        img_pil = transforms.ToPILImage()(img)
        img_save_path = os.path.join('validation_samples', f'val_img_{i}.png')
        img_pil.save(img_save_path)
    print(f"Saved {len(validation_dataset)} validation images to 'validation_samples' folder.", flush=True)
    # Count the number of samples in each class
    total_counts = Counter()

    for _, labels in train_dataset:
        total_counts.update(labels.flatten().tolist())

    print("Label frequency across all classes:", dict(total_counts))
    # Count the number of samples in each class for the validation dataset
    val_counts = Counter()
    for _, labels in validation_dataset:
        val_counts.update(labels.flatten().tolist())
    print("Label frequency in validation dataset:", dict(val_counts))
    
    # Print the size of an individual image from the training dataset
    print(f"Size of an individual image from the training dataset: {train_dataset[0][0].size()}", flush=True)
    print(f"Size of an individual image from the validation dataset: {validation_dataset[0][0].size()}", flush=True)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)
    print(f"Number of validation samples: {len(validation_dataset)}", flush=True)

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Load the true test set for predictions
    test_img_dir = './data/test/'
    test_dataset = ChocolateDataset(csv_file=None, img_path=test_img_dir, transform=None, preload=False)  # No CSV file for test set
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    # model = SimpleCNN(num_classes=13).to(device)
    # model = SimpleCNN_Light(num_classes=13).to(device)
    model = DeepResidualCNN(num_classes=13).to(device)
    
    # Calculate global weights across all classes
    count_hist = np.zeros(6)  # For counts 0–5

    for _, labels in train_dataset:
        for c in range(13):
            count_hist[labels[c]] += 1

    weights = 1.0 / (count_hist + 1e-6)
    weights = weights / weights.sum() * 6  # Normalize
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1, weight=weights)  # Use CrossEntropyLoss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Reduce the learning rate for better convergence

    # Print the number of parameters in the network
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the network: {num_params}", flush=True)
    
    def compute_loss(outputs, labels, criterion):
        """
        outputs: [B, 13, 6] — logits per class
        labels: [B, 13] — integer target per class (in [0, 5])
        criterion: nn.CrossEntropyLoss()
        """
        # print(labels.shape)
        # print(outputs.shape)
        B, C, K = outputs.shape  # B=batch, C=13 classes, K=6 bins
        loss = 0
        for c in range(C):
            # print(f"Class {c}: {labels[:, c]}", flush=True)
            # print(f"Class {c}: {outputs[:, c, :]}", flush=True)
            loss += criterion(outputs[:, c, :], labels[:, c])
        return loss / C
    
    # Add argument parsing for number of epochs
    parser = argparse.ArgumentParser(description="Train the Chocolate Classification Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    args = parser.parse_args()

    # Use the parsed number of epochs
    num_epochs = args.epochs
    
    # Initialize history lists
    loss_history = []
    f1_history = []
    val_loss_history = []
    val_f1_history = []

    # Add `leave=False` to TQDM to ensure only one progress bar is displayed
    best_f1_val = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_f1 = 0.0
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            assert labels.dtype == torch.int64 and labels.ndim == 2, "Labels must be class indices with shape [B, 13]"
            # print(f"Image shape: {images.shape}", flush=True)
            # print(f"Label shape: {labels.shape}", flush=True)
            # print(f"Labels: {labels}", flush=True)

            optimizer.zero_grad()
            outputs = model(images)
            # outputs = torch.round(outputs)
            # loss = criterion(outputs, labels.float())
            loss = compute_loss(outputs, labels, criterion)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate F1 Score for the batch
            with torch.no_grad():
                true = labels.int().cpu().numpy()
                running_f1 += calculate_f1_score(true, outputs.cpu().numpy())
            
                # Print if any predictions are negative
                # if np.any(outputs.cpu().numpy() < 0):
                #     print(f"Negative predictions found!")

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = running_f1 / len(train_loader)
        print(f"\n Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Training F1 Score: {epoch_f1:.4f}", flush=True)

        # Update training history
        loss_history.append(epoch_loss)
        f1_history.append(epoch_f1)

        # Run validation
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # outputs = torch.round(outputs)
                # loss = criterion(outputs, labels.float())
                loss = compute_loss(outputs, labels, criterion)
                val_loss += loss.item()

                # Calculate F1 Score for the validation batch
                val_f1 += calculate_f1_score(labels.cpu().numpy(), outputs.cpu().numpy())

        # Calculate validation metrics
        val_loss /= len(validation_loader)
        val_f1 /= len(validation_loader)
        if val_f1 > best_f1_val:
                    best_f1_val = val_f1
                    torch.save(model.state_dict(), 'best_model.pth')
                    print(f"Saved best model with F1: {val_f1:.4f}", flush=True)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.4f}", flush=True)

        # Update validation history
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

    # Plot training loss
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved loss_plot.png", flush=True)

    # Plot validation loss
    plt.figure()
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.savefig('validation_loss_plot.png')
    print("Saved validation_loss_plot.png", flush=True)

    # Plot F1 Score
    plt.figure()
    plt.plot(f1_history, label='Training F1 Score')
    plt.plot(val_f1_history, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score')
    plt.legend()
    plt.savefig('f1_score_plot.png')
    print("Saved f1_score_plot.png", flush=True)

    # Simplify the test set prediction logic
    model.eval()
    kaggle_predictions = []

    with torch.no_grad():
        for images, img_ids in tqdm(test_loader):  # No labels for Kaggle test set
            # print(f"Image IDs: {img_ids}", flush=True)
            images = images.to(device)
            outputs = model(images)
            # outputs = torch.round(outputs)
            preds = torch.argmax(outputs, dim=-1).cpu().numpy()  # Get predicted class indices
            # Ensure IDs in the predictions file match the format in sample_submission.csv
            image_ids = [img_id[1:] for img_id in img_ids]  # Remove the 'L' prefix from image IDs
            # print(img_ids)
            # print(preds.shape)
            for img_id, pred in zip(image_ids, preds):
                # print(f"Image ID: {img_id}, Prediction: {pred}")  # Debugging line
                kaggle_predictions.append([img_id] + pred.tolist())

    # Create a DataFrame for Kaggle test set predictions
    columns = ["id", "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée", "Triangolo", "Tentation noir", "Comtesse", "Noblesse", "Noir authentique", "Passion au lait", "Arabia", "Stracciatella"]
    kaggle_predictions_df = pd.DataFrame(kaggle_predictions, columns=columns)

    # Save Kaggle test set predictions to CSV
    kaggle_predictions_df.to_csv("sample_submission.csv", index=False)
    print("Saved kaggle_predictions.csv", flush=True)

    # Save the final trained model to the 'models' directory after all epochs
    final_model_save_path = os.path.join('models', 'final_model.pth')
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Saved final model to {final_model_save_path}", flush=True)

if __name__ == "__main__":
    main()