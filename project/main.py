import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from src.dataset import ChocolateDataset
from src.model import SimpleCNN
from tqdm import tqdm
import numpy as np

def accuracy_fn(outputs, labels):
    preds = torch.round(outputs)
    correct = (preds == labels).float().sum()
    total = torch.numel(labels)
    return correct / total

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_csv = './data/train.csv'
    train_img_dir = './data/train'

    train_dataset = ChocolateDataset(train_csv, train_img_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN(num_classes=13).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    loss_history = []
    acc_history = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_acc += accuracy_fn(outputs.detach().cpu(), labels.cpu()).item()

            # Print predictions and ground truth for the first batch of each epoch
            if batch_idx == 0:
                preds = torch.round(outputs.detach().cpu()).numpy().astype(int)
                trues = labels.cpu().numpy().astype(int)
                print(f"\nEpoch {epoch+1} - First batch predictions vs ground truth:")
                for i in range(min(3, preds.shape[0])):  # print up to 3 samples
                    print(f"Pred: {preds[i]} | True: {trues[i]}")

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / len(train_loader)
        loss_history.append(avg_loss)
        acc_history.append(avg_acc)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

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

if __name__ == "__main__":
    main()