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

# Update accuracy function to calculate exact match accuracy
def accuracy_fn(outputs, labels):
    preds = torch.round(outputs).int()  # Round predictions to nearest integer
    correctly_predicted = (preds == labels).all(dim=1)  # Check if all elements in a row match
    # print(f"Predictions: {preds}, Labels: {labels}")  # Debugging line
    # print(f"Correctly predicted: {correctly_predicted}")  # Debugging line
    accuracy = correctly_predicted.sum().item() / len(labels)  # Calculate percentage of exact matches
    return accuracy

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_csv = './data/train.csv'
    train_img_dir = './data/train'

    # Print when data has been loaded
    print("Loading training data...")
    train_dataset = ChocolateDataset(train_csv, train_img_dir, transform = transform)
    print("Training data loaded.")

    # Split train and test datasets
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    print("Splitting data into training and testing sets...")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print("Data split completed.")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=13).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 3
    loss_history = []
    acc_history = []
    # f1_history = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        # running_f1 = 0.0
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            # Ensure labels are converted to float before computing loss
            labels = labels.float()
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_acc += accuracy_fn(outputs.detach().cpu(), labels.cpu())

            # Calculate F1 Score for the batch     
            assert outputs.shape[0] == labels.shape[0], "Batch size mismatch"
            # running_f1 += f1_score(true, preds, average='weighted')


        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        # avg_f1 = running_f1 / len(train_loader)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        # f1_history.append(avg_f1)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}") #, F1 Score: {avg_f1:.4f}")

    plt.figure()
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    print("Saved loss_plot.png")

    plt.figure()
    plt.plot(acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    print("Saved accuracy_plot.png")

    # # Plot F1 Score
    # plt.figure()
    # plt.plot(f1_history, label='F1 Score')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1 Score')
    # plt.title('Training F1 Score')
    # plt.legend()
    # plt.savefig('f1_score_plot.png')
    # print("Saved f1_score_plot.png")

    # Update evaluation logic to handle counts for multi-label predictions
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.clamp(torch.round(outputs).int(), min=0).cpu().numpy()  # Clamp to ensure no negative values
            true = labels.cpu().numpy().astype(int)  # Ensure true labels are integers
            predictions.extend(preds)
            true_labels.extend(true)

    # Adjust visualization to handle integer counts for multi-output regression
    plt.figure(figsize=(10, 6))
    for i in range(len(predictions[0])):  # Iterate over each label
        sns.histplot([pred[i] for pred in predictions], kde=False, label=f'Predicted Count for Label {i}', alpha=0.5)
        sns.histplot([true[i] for true in true_labels], kde=False, label=f'True Count for Label {i}', alpha=0.5)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Predicted vs True Count Distribution (Multi-Output Regression)')
    plt.legend()
    plt.savefig('multi_output_count_distribution.png')
    print("Saved multi_output_count_distribution.png")

    # Save predictions to CSV in the same format as train.csv
    predictions = []
    # Ensure image_ids matches the number of predictions
    image_ids = [f"L{img_id}" for img_id in test_loader.dataset.indices]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.clamp(torch.round(outputs).int(), min=0).cpu().numpy()  # Clamp to ensure no negative values
            predictions.extend(preds)

    # Create a DataFrame for predictions
    columns = ["id", "Jelly White", "Jelly Milk", "Jelly Black", "Amandina", "Crème brulée", "Triangolo", "Tentation noir", "Comtesse", "Noblesse", "Noir authentique", "Passion au lait", "Arabia", "Stracciatella"]
    predictions_df = pd.DataFrame(predictions, columns=columns[1:])
    predictions_df.insert(0, "id", image_ids)

    # Save to CSV
    predictions_df.to_csv("test_predictions.csv", index=False)
    print("Saved test_predictions.csv")

    # Calculate test set accuracy
    model.eval()
    test_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_accuracy += accuracy_fn(outputs, labels)
            num_batches += 1

    test_accuracy /= num_batches

    # Save test set accuracy to a text file
    with open("test_accuracy.txt", "w") as f:
        f.write(f"Test Set Accuracy: {test_accuracy:.4f}\n")
    print("Saved test_accuracy.txt")

if __name__ == "__main__":
    main()