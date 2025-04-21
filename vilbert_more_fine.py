import os
import random
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import timm
from torchvision import models, transforms
from transformers import ViltModel, ViltProcessor

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# -----------------------------------------
# Global image transform (with augmentation)
# -----------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# -----------------------------------------
# Dataset
# -----------------------------------------
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, processor, split='train', train_ratio=1.0):
        self.images_path = images_path
        self.processor = processor
        self.samples = []
        with open(data_file, 'r') as f:
            lines = f.readlines()
        if split == 'train':
            random.shuffle(lines)
            lines = lines[:int(len(lines) * train_ratio)]
        for line in lines:
            img_name, qa_label, lbl = line.strip().split('\t')
            q, a = qa_label.split('?', 1)
            q = q.strip() + '?'
            a = a.strip()
            text = q + ' ' + a
            label = 1 if lbl.strip() == 'match' else 0
            self.samples.append((img_name, text, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, text, label = self.samples[idx]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        return image, text, torch.tensor(label, dtype=torch.long)

# -----------------------------------------
# Encoders (freeze all except last blocks)
# -----------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        m = models.resnet50(pretrained=True)
        for p in m.parameters(): p.requires_grad = False
        for p in m.layer4.parameters(): p.requires_grad = True
        m.fc = nn.Linear(m.fc.in_features, embed_dim)
        self.model = m

    def forward(self, x):
        return self.model(x)

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        v = timm.create_model('vit_base_patch16_224', pretrained=True)
        for p in v.parameters(): p.requires_grad = False
        for p in v.blocks[-1].parameters(): p.requires_grad = True
        v.reset_classifier(0)
        self.model = v
        self.proj = nn.Linear(v.num_features, embed_dim)

    def forward(self, x):
        return self.proj(self.model(x))

class ViLTTextEncoder(nn.Module):
    def __init__(self, model_name='dandelin/vilt-b32-mlm', embed_dim=128):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.vilt = ViltModel.from_pretrained(model_name)
        for p in self.vilt.parameters(): p.requires_grad = False
        # unfreeze final fusion layer
        for p in self.vilt.encoder.layer[-1].parameters(): p.requires_grad = True
        self.proj = nn.Linear(self.vilt.config.hidden_size, embed_dim)

    def forward(self, image, text, device):
        inputs = self.processor(image, text,
                                return_tensors='pt',
                                padding=True,
                                truncation=True).to(device)
        out = self.vilt(**inputs)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)

# -----------------------------------------
# Attention Fusion
# -----------------------------------------
class AttentionFusionModule(nn.Module):
    def __init__(self, embed_dim, heads=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=True)

    def forward(self, f1, f2):
        stacked = torch.stack([f1, f2], dim=1)
        attn_out, _ = self.mha(stacked, stacked, stacked)
        return attn_out.mean(dim=1)

# -----------------------------------------
# Ensemble Model: ResNet + ViT + ViLT
# -----------------------------------------
class EnsembleFusionModel(nn.Module):
    def __init__(self, embed_dim=128, num_classes=2):
        super().__init__()
        self.resnet = ResNetEncoder(embed_dim)
        self.vit    = ViTEncoder(embed_dim)
        self.fusion = AttentionFusionModule(embed_dim)
        self.text   = ViLTTextEncoder(embed_dim=embed_dim)
        # classifier: concat(image_fused (embed_dim), text (embed_dim)) => 2*embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, text, device):
        img = image_transform(image).unsqueeze(0).to(device)
        f1  = self.resnet(img)
        f2  = self.vit(img)
        f_img = self.fusion(f1, f2)
        f_txt = self.text(image, text, device)
        feat  = torch.cat([f_img, f_txt], dim=1)
        return self.classifier(feat)

# -----------------------------------------
# Training & Validation
# -----------------------------------------
def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    tot_loss = 0.0
    tot_acc  = 0
    tot_n    = 0

    for im, txt, lbl in loader:
        im_t = image_transform(im).unsqueeze(0).to(device)
        inputs = model.text.processor(im, txt,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True).to(device)
        y  = lbl.to(device).unsqueeze(0)

        # MixUp
        lam = np.random.beta(0.2, 0.2)
        idx = torch.randperm(im_t.size(0))
        im2 = im_t[idx]; y2 = y[idx]
        im_mix = lam*im_t + (1-lam)*im2

        opt.zero_grad()
        f1 = model.resnet(im_mix)
        f2 = model.vit(im_mix)
        f_img = model.fusion(f1, f2)
        f_txt = model.text(im, txt, device)
        feat  = torch.cat([f_img, f_txt], dim=1)
        logits = model.classifier(feat)

        loss = lam*crit(logits, y) + (1-lam)*crit(logits, y2)
        loss.backward()
        opt.step()

        preds = logits.argmax(dim=1)
        tot_loss += loss.item() * y.size(0)
        tot_acc  += (preds == y).sum().item()
        tot_n    += y.size(0)

    return tot_loss/tot_n, tot_acc/tot_n

def validate(model, loader, crit, device):
    model.eval()
    tot_loss = 0.0
    tot_acc  = 0
    tot_n    = 0
    ys, ps, pr = [], [], []

    with torch.no_grad():
        for im, txt, lbl in loader:
            im_t = image_transform(im).unsqueeze(0).to(device)
            inputs = model.text.processor(im, txt,
                                          return_tensors='pt',
                                          padding=True,
                                          truncation=True).to(device)
            y  = lbl.to(device).unsqueeze(0)

            f1 = model.resnet(im_t)
            f2 = model.vit(im_t)
            f_img = model.fusion(f1, f2)
            f_txt = model.text(im, txt, device)
            feat  = torch.cat([f_img, f_txt], dim=1)
            logits = model.classifier(feat)

            loss = crit(logits, y)
            preds = logits.argmax(dim=1)
            prob  = torch.softmax(logits, dim=1)[0,1].item()

            tot_loss += loss.item() * y.size(0)
            tot_acc  += (preds == y).sum().item()
            tot_n    += y.size(0)

            ys.append(y.item())
            ps.append(preds.item())
            pr.append(prob)

    return tot_loss/tot_n, tot_acc/tot_n, ys, ps, pr

# -----------------------------------------
# Plotting Utilities
# -----------------------------------------
def plot_training(tl, vl, ta, va):
    epochs = range(1, len(tl)+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, tl, '-o', label='Train Loss')
    plt.plot(epochs, vl, '-o', label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.legend(); plt.grid()
    plt.subplot(1,2,2)
    plt.plot(epochs, ta, '-o', label='Train Acc')
    plt.plot(epochs, va, '-o', label='Val Acc')
    plt.title('Accuracy Curve'); plt.xlabel('Epoch'); plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig('training_validation_curves.png')
    plt.close()

def plot_conf(ys, ps):
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix'); plt.colorbar()
    ticks = [0,1]
    plt.xticks(ticks,['no','yes']); plt.yticks(ticks,['no','yes'])
    thresh = cm.max()/2
    for i,j in np.ndindex(cm.shape):
        plt.text(j,i,cm[i,j],
                 ha='center',
                 color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_roc_curve(ys, pr):
    fpr, tpr, _ = roc_curve(ys, pr)
    auc = roc_auc_score(ys, pr)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    plt.plot([0,1],[0,1],'--')
    plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

# -----------------------------------------
# Main
# -----------------------------------------
if __name__ == '__main__':
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')

    img_dir   = 'visual7w-images'
    train_txt = 'visual7w-text/v7w.TrainImages.itm.txt'
    dev_txt   = 'visual7w-text/v7w.DevImages.itm.txt'
    test_txt  = 'visual7w-text/v7w.TestImages.itm.txt'

    train_ds = ITM_Dataset(img_dir, train_txt, processor, split='train', train_ratio=0.3)
    dev_ds   = ITM_Dataset(img_dir, dev_txt, processor, split='dev')
    test_ds  = ITM_Dataset(img_dir, test_txt, processor, split='test')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,  collate_fn=lambda x: x[0])
    dev_loader   = DataLoader(dev_ds,   batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # build model
    model = EnsembleFusionModel().to(device)

    # layer‑wise LRs
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'resnet' in name or 'vit' in name or 'vilt' in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params,     'lr': 3e-5},
    ], weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    epochs = 5
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(1, epochs+1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, ys_dev, ps_dev, pr_dev = validate(model, dev_loader, criterion, device)

        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"Epoch {ep}/{epochs}: "
              f"Train Loss {tl:.4f}, Acc {ta:.4f} | "
              f"Val   Loss {vl:.4f}, Acc {va:.4f}")

    # threshold sweep on validation set
    best_thr, best_f1 = 0.0, 0.0
    for thr in np.linspace(0, 1, 51):
        preds_thr = [1 if p > thr else 0 for p in pr_dev]
        f1 = f1_score(ys_dev, preds_thr)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"Optimal threshold: {best_thr:.2f}, Validation F1: {best_f1:.4f}")

    # final plots
    plot_training(train_losses, val_losses, train_accs, val_accs)
    plot_conf(ys_dev, ps_dev)
    plot_roc_curve(ys_dev, pr_dev)

    # test set evaluation
    ys_test, ps_test, pr_test = validate(model, test_loader, criterion, device)[2:]
    print("Test Accuracy :", accuracy_score(ys_test, ps_test))
    print("Precision     :", precision_score(ys_test, ps_test))
    print("Recall        :", recall_score(ys_test, ps_test))
    print("F1 Score      :", f1_score(ys_test, ps_test))
    print("AUC           :", roc_auc_score(ys_test, pr_test))
    print("Done.")
