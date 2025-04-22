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
# Flags to toggle various options:
USE_ALT_TEXT_ENCODER = True           # If True, use DistilBERT for text encoding (fine-tuned)
USE_FOCAL_LOSS = True                  # If True, use Focal Loss instead of CrossEntropyLoss

###########################################
# Focal Loss (for class imbalance)
###########################################
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a scalar (e.g., 0.75) or a tensor for class weights
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            # If alpha is provided, assume binary classification, create tensor if needed.
            if isinstance(self.alpha, (float, int)):
                alpha = torch.tensor([self.alpha, 1 - self.alpha]).to(inputs.device)
            else:
                alpha = self.alpha.to(inputs.device)
            at = alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

###########################################
# Alternative Text Encoder Module using DistilBERT with Dropout
###########################################
class AlternativeTextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", projection_dim=128, fine_tune=True, dropout_rate=0.3):
        super(AlternativeTextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        if not fine_tune:
            for param in self.text_model.parameters():
                param.requires_grad = False
    def forward(self, text_list):
        encoding = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=64)
        encoding = {key: val.to(next(self.text_model.parameters()).device) for key, val in encoding.items()}
        outputs = self.text_model(**encoding)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_output)
        projected = self.dropout(projected)
        return projected

###########################################
# Custom Dataset
###########################################
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0, use_alt_text_encoder=True):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0
        self.use_alt_text_encoder = use_alt_text_encoder

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
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
                if not self.use_alt_text_encoder:
                    self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                    self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
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
        if self.use_alt_text_encoder:
            q_feature = question_text  # return raw text
            a_feature = self.answer_data[idx]
        else:
            q_feature = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
            a_feature = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        return img, q_feature, a_feature, label, img_name, question_text
    
# Load sentence embeddings from an existing file -- generated a priori
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data 

###########################################
# Vision Encoder: Transformer_VisionEncoder using vit_base_patch16_224 (ViT-B/16) with Dropout
###########################################
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None, dropout_rate=0.3):
        super(Transformer_VisionEncoder, self).__init__()
        # Using timm to load ViT-B/16 instead of vit_b_32.
        if pretrained:
            self.vision_model = timm.create_model('vit_base_patch16_224', pretrained=True)
        else:
            self.vision_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        # Remove the classification head.
        self.vision_model.reset_classifier(0)
        self.num_features = self.vision_model.num_features
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        features = self.vision_model(x)
        return self.dropout(features)

###########################################
# Attention-based Fusion Module using Multihead Attention
###########################################
class AttentionFusionModule(nn.Module):
    def __init__(self, feature_dim, num_heads=2):
        super(AttentionFusionModule, self).__init__()
        # MultiheadAttention expects input shape (batch, seq_len, embed_dim) if batch_first=True
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
    def forward(self, cnn_feat, vit_feat):
        # Stack features: shape -> (batch, 2, feature_dim)
        feats = torch.stack([cnn_feat, vit_feat], dim=1)
        # Apply multi-head attention (using feats as query, key, and value)
        attn_output, attn_weights = self.mha(feats, feats, feats)
        # Pool outputs (e.g., mean over the sequence dimension)
        fused_feat = torch.mean(attn_output, dim=1)  # (batch, feature_dim)
        return fused_feat, attn_weights

###########################################
# Image-Text Matching Model with Ensemble option and Joint/Hybrid Text Encoder
###########################################
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None, use_alt_text_encoder=True, dropout_rate=0.3):
        print(f'BUILDING {ARCHITECTURE} model, pretrained={PRETRAINED}, use_alt_text_encoder={use_alt_text_encoder}')
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        self.use_alt_text_encoder = use_alt_text_encoder

        # Vision branches.
        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            # Add dropout after ResNet features; you can add it in the classifier head.
            self.dropout = nn.Dropout(p=dropout_rate)
            if PRETRAINED:
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128)
        
        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED, dropout_rate=dropout_rate)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)
        
        elif self.ARCHITECTURE == "Ensemble":
            # CNN branch.
            self.cnn_model = models.resnet18(pretrained=PRETRAINED)
            self.dropout_cnn = nn.Dropout(p=dropout_rate)
            if PRETRAINED:
                for param in self.cnn_model.parameters():
                    param.requires_grad = False
                for param in list(self.cnn_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.cnn_model.parameters():
                    param.requires_grad = True
            self.cnn_model.fc = nn.Linear(self.cnn_model.fc.in_features, 128)
            # ViT branch.
            self.vit_model = Transformer_VisionEncoder(pretrained=PRETRAINED, dropout_rate=dropout_rate)
            self.fc_vit = nn.Linear(self.vit_model.num_features, 128)
            # Use attention-based fusion.
            self.fusion = AttentionFusionModule(feature_dim=128, num_heads=2)
        
        else:
            raise ValueError("UNKNOWN neural architecture")

        # Text processing.
        if not self.use_alt_text_encoder:
            self.question_embedding_layer = nn.Linear(768, 128)
            self.answer_embedding_layer = nn.Linear(768, 128)
        else:
            self.alt_text_encoder = AlternativeTextEncoder(model_name="distilbert-base-uncased", projection_dim=128, fine_tune=True, dropout_rate=dropout_rate)
        
        # Final classification layer.
        self.fc = nn.Linear(128 + 128 + 128, num_classes)

    def forward(self, img, question_input, answer_input):
        # Vision processing.
        if self.ARCHITECTURE == "CNN":
            img_features = self.vision_model(img)
            img_features = self.dropout(img_features)
        elif self.ARCHITECTURE == "ViT":
            img_features = self.vision_model(img)
            img_features = self.fc_vit(img_features)
        elif self.ARCHITECTURE == "Ensemble":
            cnn_features = self.cnn_model(img)
            cnn_features = self.dropout_cnn(cnn_features)
            vit_features = self.vit_model(img)
            vit_features = self.fc_vit(vit_features)
            img_features, attn_weights = self.fusion(cnn_features, vit_features)
        else:
            raise ValueError("Unsupported architecture")

        # Text processing.
        if self.use_alt_text_encoder:
            question_features = self.alt_text_encoder(question_input)
            answer_features = self.alt_text_encoder(answer_input)
        else:
            question_features = self.question_embedding_layer(question_input)
            answer_features = self.answer_embedding_layer(answer_input)
        
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

###########################################
# Optimizer and Scheduler with parameter groups for text encoder
###########################################
def get_optimizer_and_scheduler(model, base_lr=3e-5, text_lr=1e-5):
    param_groups = []
    # Collect parameters not in alt_text_encoder.
    non_text_params = [p for n, p in model.named_parameters() if "alt_text_encoder" not in n]
    param_groups.append({"params": non_text_params, "lr": base_lr})
    if USE_ALT_TEXT_ENCODER:
        param_groups.append({"params": model.alt_text_encoder.parameters(), "lr": text_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
    return optimizer, scheduler

###########################################
# Early Stopping: Evaluate accuracy on validation set
###########################################
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 6:
                images, question_input, answer_input, labels, _, _ = batch
            else:
                images, question_input, answer_input, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            if not USE_ALT_TEXT_ENCODER:
                question_input = question_input.to(device)
                answer_input = answer_input.to(device)
            outputs = model(images, question_input, answer_input)
            preds = torch.softmax(outputs, dim=1).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct/total

###########################################
# Training Function with Early Stopping, Learning Rate Scheduler, and Dropout regularization
###########################################
def train_model(model, ARCHITECTURE, train_loader, val_loader, criterion, optimiser, scheduler, num_epochs=15, patience=3):
    print(f'TRAINING {ARCHITECTURE} model')
    model.train()
    overall_start = time.time()
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    # Track training and validation metrics
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if len(batch) == 6:
                images, question_input, answer_input, labels, _, _ = batch
            else:
                images, question_input, answer_input, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            if not USE_ALT_TEXT_ENCODER:
                question_input = question_input.to(device)
                answer_input = answer_input.to(device)
            outputs = model(images, question_input, answer_input)
            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            
        scheduler.step()
        
        epoch_loss = running_loss / total_batches
        epoch_duration = time.time() - epoch_start
        
        # Calculate training accuracy
        train_acc = evaluate_accuracy(model, train_loader, device)
        # Calculate validation accuracy
        val_acc = evaluate_accuracy(model, val_loader, device)
        
        # Add metrics to lists
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(epoch_loss)
        val_losses.append(epoch_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f} s, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break
    
    overall_duration = time.time() - overall_start
    print(f'TOTAL Training Time: {overall_duration:.2f} seconds')
    
    # Plot training and validation curves
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - {ARCHITECTURE}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'g-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - {ARCHITECTURE}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'distilbert_training_curves_{ARCHITECTURE}.png')
    plt.close()

###########################################
# Evaluation Function with additional metrics (Precision, Recall, F1-Score, AUC)
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
                images, question_input, answer_input, labels, img_names, question_texts = batch
            else:
                images, question_input, answer_input, labels = batch
                img_names = ["unknown"] * images.size(0)
                question_texts = ["unknown"] * images.size(0)
            images = images.to(device)
            labels = labels.to(device)
            if not USE_ALT_TEXT_ENCODER:
                question_input = question_input.to(device)
                answer_input = answer_input.to(device)
            outputs = model(images, question_input, answer_input)
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

    # Compute additional metrics
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    auc_value = roc_auc_score(all_labels, all_probabilities)

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
    print(f'Balanced Accuracy: {test_accuracy:.4f}')
    print(f'Total Test Loss: {test_loss:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'AUC: {auc_value:.4f}')
    print(f'Testing Time: {eval_duration:.2f} seconds')

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Non-Match', 'Match'], 
               yticklabels=['Non-Match', 'Match'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {ARCHITECTURE}')
    plt.savefig(f'distilbert_cm_{ARCHITECTURE}.png')
    plt.close()

    # Generate and save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {ARCHITECTURE}')
    plt.legend(loc="lower right")
    plt.savefig(f'distilbert_roc_{ARCHITECTURE}.png')
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
    sentence_embeddings_file = "/workspaces/ADV-ML/v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)
    
    # Create datasets and loaders.
    # For early stopping, we'll use the dev set. If not available, use test as a proxy.
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2, use_alt_text_encoder=USE_ALT_TEXT_ENCODER)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    dev_dataset = ITM_Dataset(IMAGES_PATH, dev_data_file, sentence_embeddings, data_split="dev", use_alt_text_encoder=USE_ALT_TEXT_ENCODER)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test", use_alt_text_encoder=USE_ALT_TEXT_ENCODER)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Choose architecture: "CNN", "ViT", or "Ensemble"
    MODEL_ARCHITECTURE = "Ensemble"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL, use_alt_text_encoder=USE_ALT_TEXT_ENCODER, dropout_rate=0.3).to(device)
    print("\nModel Architecture:")
    print(model)
    
    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters: {total_params}")
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}, USE_ALT_TEXT_ENCODER={USE_ALT_TEXT_ENCODER}\n")
    
    # Choose loss function.
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(gamma=2, alpha=0.75, reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimiser, scheduler = get_optimizer_and_scheduler(model, base_lr=3e-5, text_lr=1e-5)
    
    # Train with early stopping.
    train_model(model, MODEL_ARCHITECTURE, train_loader, dev_loader, criterion, optimiser, scheduler, num_epochs=15, patience=3)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
