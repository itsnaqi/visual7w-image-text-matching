import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    ViTImageProcessor,
    ViTModel
)

# Custom Dataset
class VQADataset(Dataset):
    def __init__(self, images_path, data_file, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.label_data = []

        if self.data_split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.load_data()

    def load_data(self):
        print(f"LOADING data from {self.data_file}")
        random.seed(42)
        with open(self.data_file, 'r') as f:
            lines = f.readlines()

        if self.data_split == "train":
            random.shuffle(lines)
            lines = lines[:int(len(lines) * self.train_ratio)]

        for line in lines:
            img_name, text, raw_label = line.strip().split("\t")
            img_path = os.path.join(self.images_path, img_name.strip())
            
            # Safely split question and answers at the first occurrence of "?"
            if "?" in text:
                question, answers = text.split("?", 1)
                question = question.strip() + "?"
                answer_list = answers.strip().split(" | ")
            else:
                # If no "?", we can handle this case or skip this line
                continue  # Or you can assign default values if needed

            correct = answer_list[0].strip()

            for ans in answer_list:
                self.image_data.append(img_path)
                self.question_data.append(question)
                self.answer_data.append(ans.strip())
                self.label_data.append(1 if ans.strip() == correct else 0)

        print(f"|image_data|={len(self.image_data)}")
        print(f"|question_data|={len(self.question_data)}")
        print(f"|answer_data|={len(self.answer_data)}")
        print("done loading data...")
        
    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img = Image.open(self.image_data[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.question_data[idx], self.answer_data[idx], torch.tensor(self.label_data[idx], dtype=torch.long)

# BLIP + ViT Model
class BLIP_ViT_Ensemble(nn.Module):
    def __init__(self, num_classes=4):
        super(BLIP_ViT_Ensemble, self).__init__()
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

        self.vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        for name, param in self.blip_model.named_parameters():
            if "vqa_head" not in name:
                param.requires_grad = False

        for name, param in self.vit_model.named_parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(768 + 768, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, images, questions, answers=None, labels=None):
        pil_images = [to_pil_image(img.cpu()) for img in images]

        # Ensure batch-safe
        if isinstance(questions, str):
            questions = [questions]
        elif isinstance(questions, list) and isinstance(questions[0], str):
            pass
        else:
            questions = [str(q) for q in questions]

        blip_inputs = self.blip_processor(images=pil_images, text=questions, return_tensors="pt", padding=True).to(images.device)
        blip_outputs = self.blip_model(**blip_inputs, labels=labels)
        blip_features = blip_outputs.logits

        vit_inputs = self.vit_processor(images=pil_images, return_tensors="pt").to(images.device)
        vit_outputs = self.vit_model(**vit_inputs)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]

        combined = torch.cat((blip_features, vit_features), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)

        return output, blip_outputs.loss

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, questions, answers, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, blip_loss = model(images, questions, answers, labels)
            loss = criterion(outputs, labels) + blip_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = validate_model(model, val_loader, device)
        model.train()

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break

def validate_model(model, val_loader, device):
    model.eval()
    total_loss, total_samples, mrr, em = 0.0, 0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, questions, answers, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images, questions, answers, labels)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            sorted_idx = torch.argsort(probs, dim=1, descending=True)
            for i in range(len(labels)):
                rank = (sorted_idx[i] == labels[i]).nonzero().item() + 1
                mrr += 1.0 / rank
                em += 1.0 if rank == 1 else 0.0

    print(f"Validation Accuracy: {em/total_samples:.4f}, Loss: {total_loss/total_samples:.4f}, MRR: {mrr/total_samples:.4f}")
    return total_loss / total_samples

def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total, mrr, em = 0, 0, 0.0, 0.0

    with torch.no_grad():
        for images, questions, answers, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images, questions, answers)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            sorted_idx = torch.argsort(probs, dim=1, descending=True)
            for i in range(len(labels)):
                rank = (sorted_idx[i] == labels[i]).nonzero().item() + 1
                mrr += 1.0 / rank
                em += 1.0 if rank == 1 else 0.0

    print(f"Test Accuracy: {correct / total:.4f}, MRR: {mrr / total:.4f}, EM: {em / total:.4f}")

if __name__ == "__main__":
    # Paths and hyperparameters
    IMAGES_PATH = "./visual7w-images"
    train_data_file = "./visual7w-text/v7w.TrainImages.itm.txt"
    val_data_file = "./visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "./visual7w-text/v7w.TestImages.itm.txt"

    train_ratio = 0.8
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    patience = 3
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VQADataset(IMAGES_PATH, train_data_file, "train", train_ratio)
    val_dataset = VQADataset(IMAGES_PATH, val_data_file, "val")
    test_dataset = VQADataset(IMAGES_PATH, test_data_file, "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BLIP_ViT_Ensemble(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience)
    evaluate_model(model, test_loader, device)
    print("Training + Evaluation time:", time.time() - start_time)
