import os
import random
import time
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
from PIL import Image
import timm
from torchvision import models, transforms
from transformers import ViltModel, ViltProcessor
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# -----------------------------------------
# Dataset
# -----------------------------------------
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, processor, split='train', train_ratio=1.0):
        self.images_path = images_path
        self.samples = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
        if split == 'train':
            random.shuffle(lines)
            lines = lines[: int(len(lines) * train_ratio)]
        for line in lines:
            img_name, qa_label, lbl = line.strip().split('\t')
            q, a = qa_label.split('?', 1)
            q = q.strip() + '?'
            a = a.strip()
            text = q + ' ' + a
            label = 1 if lbl.strip() == 'match' else 0
            self.samples.append((img_name, text, label))
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text, label = self.samples[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        return image, text, torch.tensor(label, dtype=torch.long)

# -----------------------------------------
# Encoders (frozen)
# -----------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        model = models.resnet50(pretrained=True)
        for p in model.parameters(): p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, embed_dim)
        self.model = model

    def forward(self, x):
        return self.model(x)

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for p in vit.parameters(): p.requires_grad = False
        vit.reset_classifier(0)
        self.model = vit
        self.proj = nn.Linear(vit.num_features, embed_dim)

    def forward(self, x):
        return self.proj(self.model(x))

class ViLTTextEncoder(nn.Module):
    def __init__(self, model_name='dandelin/vilt-b32-mlm', embed_dim=128):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.vilt = ViltModel.from_pretrained(model_name)
        for p in self.vilt.parameters(): p.requires_grad = False
        hidden_size = self.vilt.config.hidden_size
        self.proj = nn.Linear(hidden_size, embed_dim)

    def forward(self, image, text, device):
        inputs = self.processor(image, text, return_tensors='pt').to(device)
        outputs = self.vilt(**inputs)
        cls = outputs.last_hidden_state[:, 0, :]
        return self.proj(cls)

# -----------------------------------------
# Attention Fusion
# -----------------------------------------
class AttentionFusionModule(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, feat1, feat2):
        stacked = torch.stack([feat1, feat2], dim=1)  # [B, 2, D]
        attn_out, _ = self.mha(stacked, stacked, stacked)
        return attn_out.mean(dim=1)

# -----------------------------------------
# Ensemble Model: ResNet + ViT + ViLT
# -----------------------------------------
class EnsembleFusionModel(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.resnet = ResNetEncoder(embed_dim)
        self.vit = ViTEncoder(embed_dim)
        self.fusion = AttentionFusionModule(embed_dim)
        self.text_encoder = ViLTTextEncoder(embed_dim=embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text, device):
        # Image transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)
        f_res = self.resnet(img_tensor)
        f_vit = self.vit(img_tensor)
        f_img = self.fusion(f_res, f_vit)
        f_txt = self.text_encoder(image, text, device)
        feat = torch.cat([f_img, f_txt], dim=1)
        return self.classifier(feat)

# -----------------------------------------
# Training & Evaluation
# -----------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img, txt, lbl in loader:
        optimizer.zero_grad()
        out = model(img, txt, device)
        loss = criterion(out, lbl.to(device).unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for img, txt, lbl in loader:
            out = model(img, txt, device)
            p = torch.softmax(out, dim=1)
            ys.append(lbl.item())
            preds.append(p.argmax(dim=1).item())
            probs.append(p[0, 1].item())
    return ys, preds, probs

# -----------------------------------------
# Plotting Utilities
# -----------------------------------------
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.title('Accuracy Curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid()

    plt.tight_layout()
    plt.savefig('training_validation_curves.png')
    plt.close()


def plot_confusion(ys, preds):
    cm = confusion_matrix(ys, preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, ['no-match', 'match']); plt.yticks(ticks, ['no-match', 'match'])
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_roc(ys, probs):
    fpr, tpr, _ = roc_curve(ys, probs)
    auc_val = roc_auc_score(ys, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], '--', label='Chance')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve'); plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

# -----------------------------------------
# Main
# -----------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')

    img_dir = 'visual7w-images'
    train_txt = 'visual7w-text/v7w.TrainImages.itm.txt'
    dev_txt = 'visual7w-text/v7w.DevImages.itm.txt'
    test_txt = 'visual7w-text/v7w.TestImages.itm.txt'

    train_ds = ITM_Dataset(img_dir, train_txt, processor, split='train', train_ratio=0.2)
    dev_ds = ITM_Dataset(img_dir, dev_txt, processor, split='dev')
    test_ds = ITM_Dataset(img_dir, test_txt, processor, split='test')

    # Collate to single sample
    collate_fn = lambda batch: batch[0]
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = EnsembleFusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=3e-4)

    epochs = 10
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(epochs):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device)
        ys_dev, preds_dev, probs_dev = evaluate(model, dev_loader, device)
        ta = accuracy_score(ys_dev, preds_dev)
        train_losses.append(tl)
        # Use dev loss for illustration
        # Compute validation loss on first dev sample for illustration
        img_dev, txt_dev, lbl_dev = dev_ds[0]
        out_dev = model(img_dev, txt_dev, device)
        val_loss = criterion(out_dev, torch.tensor([lbl_dev], device=device)).item()
        val_losses.append(val_loss)
        train_accs.append(ta)
        val_accs.append(ta)
        print(f'Epoch {ep+1}/{epochs} - Train Loss: {tl:.4f}, Val Acc: {ta:.4f}')

    # Plot curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Final evaluation on test set
    ys_test, preds_test, probs_test = evaluate(model, test_loader, device)
    print('Test Acc:', accuracy_score(ys_test, preds_test))
    plot_confusion(ys_test, preds_test)
    plot_roc(ys_test, probs_test)
    print(f"Precision: {precision_score(ys_test, preds_test):.4f}")
    print(f"Recall:    {recall_score(ys_test, preds_test):.4f}")
    print(f"F1‑Score:  {f1_score(ys_test, preds_test):.4f}")
    print(f"AUC:       {roc_auc_score(ys_test, probs_test):.4f}")
    print('Done.')
