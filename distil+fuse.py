import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import timm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

###########################################
# Alternative Text Encoder Module using DistilBERT (frozen)
###########################################
class AlternativeTextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", projection_dim=128):
        super(AlternativeTextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        # Freeze all text model parameters
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.projection = nn.Linear(768, projection_dim)
    
    def forward(self, text_list):
        encoding = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64)
        encoding = {key: val.to(next(self.text_model.parameters()).device) for key, val in encoding.items()}
        with torch.no_grad():
            outputs = self.text_model(**encoding)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_output)
        return projected

###########################################
# Custom Dataset
###########################################
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.load_data()

    def load_data(self):
        print("LOADING data from " + self.data_file)
        random.seed(42)
        with open(self.data_file) as f:
            lines = f.readlines()
            if self.data_split == "train":
                random.shuffle(lines)
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]
            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")
                img_path = os.path.join(self.images_path, img_name.strip())
                qa = text.split("?")
                question_text = qa[0].strip() + '?'
                answer_text = qa[1].strip()
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.label_data.append(label)
        print(f"|image_data|={len(self.image_data)} | |question_data|={len(self.question_data)} | |answer_data|={len(self.answer_data)}")
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        img_name = os.path.basename(img_path)
        question_text = self.question_data[idx]
        return img, question_text, self.answer_data[idx], label, img_name, question_text

###########################################
# Vision Encoder: Transformer_VisionEncoder using vit_base_patch16_224
###########################################
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Transformer_VisionEncoder, self).__init__()
        self.vision_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vision_model.reset_classifier(0)
        # Freeze all vision model parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False
        # Get the number of features from the model
        self.num_features = self.vision_model.num_features
    
    def forward(self, x):
        features = self.vision_model(x)
        return features

###########################################
# Image-Text Matching Model with Ensemble option and Frozen Text Encoder
###########################################
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(ITM_Model, self).__init__()
        # CNN branch with frozen weights
        self.cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.cnn_model.parameters():
            param.requires_grad = False
        self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, 128)
        
        # ViT branch with frozen weights
        self.vit_model = Transformer_VisionEncoder(pretrained=True)
        self.fc_vit = nn.Linear(self.vit_model.num_features, 128)
        
        # Text processing with frozen DistilBERT
        self.alt_text_encoder = AlternativeTextEncoder()
        
        # Final classification layer
        self.fc = nn.Linear(512, num_classes)  # Corrected input dimension

    def forward(self, img, question_text, answer_text):
        # Vision processing
        cnn_features = self.cnn_model(img)
        vit_features = self.vit_model(img)
        vit_features = self.fc_vit(vit_features)
        
        # Simple feature fusion (concatenation)
        img_features = torch.cat((cnn_features, vit_features), dim=1)
        
        # Text processing
        question_features = self.alt_text_encoder(question_text)
        answer_features = self.alt_text_encoder(answer_text)
        
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output
###########################################
# Training Function with Early Stopping and Plotting
###########################################
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3):
    model.train()
    overall_start = time.time()
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for batch in train_loader:
            img, question_text, answer_text, labels, _, _ = batch
            img = img.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(img, question_text, answer_text)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                img, question_text, answer_text, labels, _, _ = batch
                img = img.to(device)
                labels = labels.to(device)
                
                outputs = model(img, question_text, answer_text)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    overall_duration = time.time() - overall_start
    print(f'TOTAL Training Time: {overall_duration:.2f} seconds')

###########################################
# Plot Training History
###########################################
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create plots
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'g-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'g-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('distilbert_training_curves.png')
    plt.close()

###########################################
# Evaluation Function with Plots
###########################################
def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            img, question_text, answer_text, labels, _, _ = batch
            img = img.to(device)
            labels = labels.to(device)
            
            outputs = model(img, question_text, answer_text)
            probabilities = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
    
    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc_value = roc_auc_score(all_labels, all_probabilities)
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC: {auc_value:.4f}')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Match', 'Match'], 
               yticklabels=['Non-Match', 'Match'])
    plt.title('Confusion Matrix')
    plt.savefig('distilbert_confusion_matrix.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('distilbert_roc_curve.png')
    plt.close()

###########################################
# Main Execution
###########################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Paths and files
    IMAGES_PATH = "/workspaces/ADV-ML/visual7w-images"
    train_data_file = "/workspaces/ADV-ML/visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "/workspaces/ADV-ML/visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "/workspaces/ADV-ML/visual7w-text/v7w.TestImages.itm.txt"
    
    # Create datasets and loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_dataset = ITM_Dataset(IMAGES_PATH, dev_data_file, data_split="dev")
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = ITM_Model(num_classes=2).to(device)
    print("\nModel Architecture:")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training
    train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=15, patience=3)
    
    # Evaluation
    evaluate_model(model, test_loader, device)