import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from src.dataset import ChocolateDataset
from src.model import SimpleCNN
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score
import pandas as pd
import os
import math
import argparse

# Function to calculate the image-wise average F1-score
def calculate_f1_score(y_true, y_pred):
    N, C = y_true.shape
    f1_scores = []

    for i in range(N):
        TP_i = sum(min(y_true[i, j], y_pred[i, j]) for j in range(C))
        FPN_i = sum(abs(y_true[i, j] - y_pred[i, j]) for j in range(C))
        if 2 * TP_i + FPN_i > 0:
            f1_scores.append((2 * TP_i) / (2 * TP_i + FPN_i))
        else:
            f1_scores.append(0.0)

    return sum(f1_scores) / N

def main():
    # Add more transforms for data augmentation
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_csv = './data/train.csv'
    train_img_dir = './data/train'

    # Print when data has been loaded
    print("Loading training data...", flush=True)
    train_dataset = ChocolateDataset(train_csv, train_img_dir, transform = transform)
    print("Training data loaded.", flush=True)

    # Re-incorporate 80/20 train-test split from the train set
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    # Print the size of an individual image from the training dataset
    print(f"Size of an individual image from the training dataset: {train_dataset[0][0].size()}", flush=True)
    print(f"Size of an individual image from the validation dataset: {validation_dataset[0][0].size()}", flush=True)
    print(f"Number of training samples: {len(train_dataset)}", flush=True)
    print(f"Number of validation samples: {len(validation_dataset)}", flush=True)

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

    # Load the true test set for predictions
    test_img_dir = './data/test/'
    test_dataset = ChocolateDataset(csv_file=None, img_path=test_img_dir, transform=transform)  # No CSV file for test set
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    model = SimpleCNN(num_classes=13).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Reduce the learning rate for better convergence

    # Print the number of parameters in the network
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the network: {num_params}", flush=True)

    # Add argument parsing for number of epochs
    parser = argparse.ArgumentParser(description="Train the Chocolate Classification Model")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    args = parser.parse_args()

    # Use the parsed number of epochs
    num_epochs = args.epochs
    
    loss_history = []
    f1_history = [] # Add F1 Score tracking during training

    # Add `leave=False` to TQDM to ensure only one progress bar is displayed
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_f1 = 0.0
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float()) # removed .float() after labels
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate F1 Score for the batch
            preds = torch.round(outputs.detach().cpu()).numpy()
            true = labels.cpu().numpy()
            running_f1 += calculate_f1_score(true, preds)
            
            # Print if any predictions are negative
            if np.any(preds < 0):
                print(f"Negative predictions found!")

        epoch_loss = running_loss / len(train_loader)
        epoch_f1 = running_f1 / len(train_loader)
        print(f"\n Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}, Training F1 Score: {epoch_f1:.4f}", flush=True)

        # Run validation every 5 epochs
        # if (epoch + 1) % 5 == 0:
        #     model.eval()
        #     val_loss = 0.0
        #     with torch.no_grad():
        #         for images, labels in validation_loader:
        #             images, labels = images.to(device), labels.to(device)
        #             outputs = model(images)
        #             loss = criterion(outputs, labels.float())
        #             val_loss += loss.item()
        #     print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(validation_loader):.4f}", flush=True)
        
        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                # Calculate F1 Score for the validation batch
                preds = torch.round(outputs.cpu()).numpy()
                true = labels.cpu().numpy()
                val_f1 += calculate_f1_score(true, preds)
        val_loss /= len(validation_loader)
        val_f1 /= len(validation_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation F1 Score: {val_f1:.4f}", flush=True)

    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved loss_plot.png", flush=True)
    
    # Plot validation loss
    plt.figure()
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.savefig('validation_loss_plot.png')
    print("Saved validation_loss_plot.png", flush=True)

    # Plot F1 Score
    plt.figure()
    plt.plot(f1_history, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training F1 Score')
    plt.legend()
    plt.savefig('f1_score_plot.png')
    print("Saved f1_score_plot.png", flush=True)
    
        # Calculate F1 Score for the validation set
    validation_predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.round(outputs).int().cpu().numpy()  # Round predictions to nearest integer
            true = labels.cpu().numpy().astype(int)  # Ensure true labels are integers
            validation_predictions.extend(preds)
            true_labels.extend(true)

    # Calculate F1 Score for the validation set
    y_true = np.array(true_labels)
    y_pred = np.array(validation_predictions)
    validation_f1_score = calculate_f1_score(y_true, y_pred)

    # Save validation F1 Score to a text file
    print("Saving validation F1 Score...", flush=True)
    with open("validation_f1_score.txt", "w") as f:
        f.write(f"Validation Set F1 Score: {validation_f1_score:.4f}\n")
    print("Saved validation_f1_score.txt", flush=True)

    # Simplify the test set prediction logic
    model.eval()
    kaggle_predictions = []

    with torch.no_grad():
        for images, img_ids in test_loader:  # No labels for Kaggle test set
            images = images.to(device)
            outputs = model(images)
            preds = torch.round(outputs).int().cpu().numpy()  # Round predictions to nearest integer
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