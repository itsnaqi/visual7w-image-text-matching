''' Below is an updated version of your ensemble branch that replaces simple concatenation with a learned weighted fusion.
In this implementation, a new module called FusionModule computes gating weights based on the concatenation of the CNN and
ViT feature vectors, then fuses them via a weighted sum. The rest of the model 
(text projection layers and final classification head) remains the same, 
and all evaluation metrics are still computed as before. '''

import os
import time
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32  
import timm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
###########################################
# Custom Dataset: returns 6 items per sample
###########################################
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from " + str(self.data_file))
        print("=========================================")
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

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()
                label = 1 if raw_label == "match" else 0

                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

        print("|image_data|=" + str(len(self.image_data)))
        print("|question_data|=" + str(len(self.question_data)))
        print("|answer_data|=" + str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        # Include additional keys for evaluation (MRR)
        img_name = os.path.basename(img_path)
        question_text = self.question_data[idx]
        return img, question_embedding, answer_embedding, label, img_name, question_text

###########################################
# Load sentence embeddings from file
###########################################
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

###########################################
# Vision Encoder: Transformer_VisionEncoder (using vit_b_32)
###########################################
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()
        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            for param in self.vision_model.parameters():
                param.requires_grad = False
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)
        self.num_features = self.vision_model.heads[0].in_features
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        features = self.vision_model(x)
        return features

###########################################
# Fusion Module: Weighted Fusion using a gating mechanism
###########################################
# class FusionModule(nn.Module):
#     def __init__(self, feature_dim):
#         super(FusionModule, self).__init__()
#         # Input is concatenation of two branches, so 2*feature_dim, and output 2 values (gating weights)
#         self.fc = nn.Linear(2 * feature_dim, 2)
#     def forward(self, cnn_feat, vit_feat):
#         combined = torch.cat((cnn_feat, vit_feat), dim=1)  # shape: (batch, 2*feature_dim)
#         weights = self.fc(combined)                        # shape: (batch, 2)
#         weights = torch.softmax(weights, dim=1)            # normalize weights for each sample
#         # Multiply each branch by its scalar weight and sum them (unsqueeze weights to match dimensions)
#         fused_feat = weights[:, 0].unsqueeze(1) * cnn_feat + weights[:, 1].unsqueeze(1) * vit_feat
#         return fused_feat, weights
class AdvancedFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4, dropout=0.1)
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, cnn_feat, vit_feat):
        # cnn_feat and vit_feat should be (batch_size, feature_dim)
        # Reshape to (sequence_length, batch_size, feature_dim)
        # For fusion, we can create a pseudo-sequence of length 2
        cnn_feat = cnn_feat.unsqueeze(0)  # (1, batch_size, feature_dim)
        vit_feat = vit_feat.unsqueeze(0)  # (1, batch_size, feature_dim)
        fused = torch.cat([cnn_feat, vit_feat], dim=0)  # (2, batch_size, feature_dim)
        
        attention_output, _ = self.attention(fused, fused, fused)
        
        # Take the output of both positions and combine
        combined = torch.cat([attention_output[0], attention_output[1]], dim=-1)
        return self.fc(combined)
###########################################
# Image-Text Matching Model with Ensemble option (Refined Fusion)
###########################################
###########################################
# Image-Text Matching Model with Advanced Features
###########################################
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING {ARCHITECTURE} model, pretrained={PRETRAINED}')
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE

        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet50(pretrained=PRETRAINED)  # Using ResNet50
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        
        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)
        
        elif self.ARCHITECTURE == "Ensemble":
            # CNN branch with ResNet50
            self.cnn_model = models.resnet50(pretrained=PRETRAINED)
            if PRETRAINED:
                for param in self.cnn_model.parameters():
                    param.requires_grad = False
                for param in list(self.cnn_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, 128)
            # ViT branch
            self.vit_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vit_model.num_features, 128)
            # Advanced fusion module
            self.fusion = AdvancedFusionModule(feature_dim=128)
        
        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)

        # Enhanced text projection layers with dropout and normalization
        self.question_embedding_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.answer_embedding_layer = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        if self.ARCHITECTURE == "Ensemble":
            self.fc = nn.Linear(128 + 128 + 128, num_classes)
        else:
            self.fc = nn.Linear(128 + 128 + 128, num_classes)
        
        # Regularization layers
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(128)

    def forward(self, img, question_embedding, answer_embedding):
        if self.ARCHITECTURE == "CNN":
            img_features = self.vision_model(img)
        elif self.ARCHITECTURE == "ViT":
            img_features = self.vision_model(img)
            img_features = self.fc_vit(img_features)
        elif self.ARCHITECTURE == "Ensemble":
            cnn_features = self.cnn_model(img)                     # (batch, 128)
            vit_features = self.vit_model(img)                     # (batch, num_features)
            vit_features = self.fc_vit(vit_features)               # (batch, 128)
            
            # Use the advanced fusion module
            img_features = self.fusion(cnn_features, vit_features)
            img_features = self.batch_norm(img_features)
            img_features = self.dropout(img_features)
        else:
            raise ValueError("Unsupported architecture")

        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output


###########################################
# Training Function with timing and training accuracy
###########################################
def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING {ARCHITECTURE} model')
    model.train()
    overall_start = time.time()
    train_losses = []
    train_accuracies = []
    scaler = torch.cuda.amp.GradScaler()
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=2, eta_min=1e-6)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 6:
                images, question_embeddings, answer_embeddings, labels, _, _ = batch
            else:
                images, question_embeddings, answer_embeddings, labels = batch
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)
            
            optimiser.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images, question_embeddings, answer_embeddings)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        
        scheduler.step()
        epoch_loss = running_loss / total_batches
        epoch_duration = time.time() - epoch_start

        # Compute training accuracy over training set
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                if len(batch) == 6:
                    images, question_embeddings, answer_embeddings, labels, _, _ = batch
                else:
                    images, question_embeddings, answer_embeddings, labels = batch
                images = images.to(device)
                question_embeddings = question_embeddings.to(device)
                answer_embeddings = answer_embeddings.to(device)
                labels = labels.to(device)
                outputs = model(images, question_embeddings, answer_embeddings)
                preds = torch.softmax(outputs, dim=1).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        train_accuracy = correct / total
        model.train()
        
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f}s, Training Accuracy: {train_accuracy:.4f}')
    
    overall_duration = time.time() - overall_start
    print(f'TOTAL Training Time: {overall_duration:.2f} seconds')
    
    
    # Plot training curves

def plot_training_curves(train_losses, train_accuracies, architecture):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title(f'Training Loss - {architecture}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'g-', label='Training Accuracy')
    plt.title(f'Training Accuracy - {architecture}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{architecture}.png')
    plt.close()

###########################################
# Evaluation Function: computes test loss, accuracy, and MRR with timing
###########################################
def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f'EVALUATING {ARCHITECTURE} model')
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []  # probability for class "match"
    group_keys = []         # (img_name, question_text) for MRR

    eval_start = time.time()
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 6:
                images, question_embeddings, answer_embeddings, labels, img_names, question_texts = batch
            else:
                images, question_embeddings, answer_embeddings, labels = batch
                img_names = ["unknown"] * images.size(0)
                question_texts = ["unknown"] * images.size(0)
            images = images.to(device)
            question_embeddings = question_embeddings.to(device)
            answer_embeddings = answer_embeddings.to(device)
            labels = labels.to(device)
            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            preds = probabilities.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            group_keys.extend(list(zip(img_names, question_texts)))
    
    eval_duration = time.time() - eval_start
    test_loss = total_loss / total_samples
    test_accuracy = correct / total_samples

    # Compute Mean Reciprocal Rank (MRR)
    group_dict = {}
    for key, prob, lab in zip(group_keys, all_probabilities, all_labels):
        if key not in group_dict:
            group_dict[key] = []
        group_dict[key].append((prob, lab))
    mrr_total = 0.0
    for key in group_dict:
        candidates = group_dict[key]
        sorted_candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        rank = 0
        for idx, (prob, lab) in enumerate(sorted_candidates, start=1):
            if lab == 1:
                rank = idx
                break
        if rank > 0:
            mrr_total += 1.0 / rank
    mrr = mrr_total / len(group_dict)
    
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Total Test Loss: {test_loss:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')
    print(f'Testing Time: {eval_duration:.2f} seconds')

    # Compute additional metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Match', 'Match'], 
               yticklabels=['Non-Match', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {ARCHITECTURE}')
    plt.savefig(f'confusion_matrix_{ARCHITECTURE}.png')
    plt.close()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {ARCHITECTURE}')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{ARCHITECTURE}.png')
    plt.close()


###########################################
# Main Execution
###########################################
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths and files
    IMAGES_PATH = "visual7w-images"
    train_data_file = "visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    # Create datasets and loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.3)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Choose architecture: "CNN", "ViT", or "Ensemble"
    MODEL_ARCHITECTURE = "Ensemble"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:")
    print(model)

    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters in the model: {total_params}")
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}\n")

    # Define loss function and optimiser
    # Class weights for handling imbalance
    class_weights = torch.tensor([0.3, 0.7]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # Train and evaluate the model
    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=30)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
