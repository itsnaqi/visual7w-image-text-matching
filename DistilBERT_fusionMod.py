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
from torchvision.models import vit_b_32  
import timm

# For the alternative text encoder:
from transformers import AutoTokenizer, AutoModel

# Global flag to toggle the use of the alternative text encoder.
USE_ALT_TEXT_ENCODER = True

###########################################
# Alternative Text Encoder Module using DistilBERT
###########################################
class AlternativeTextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", projection_dim=128):
        super(AlternativeTextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(768, projection_dim)
    def forward(self, text_list):
        """
        Expects text_list as a list of strings.
        Tokenizes, passes through the model, and projects the [CLS] (first) token.
        """
        # Tokenize input with padding and truncation.
        encoding = self.tokenizer(text_list, return_tensors="pt", 
                                  padding=True, truncation=True, max_length=64)
        # Send tokenized tensors to the same device as the text model's parameters.
        encoding = {key: val.to(next(self.text_model.parameters()).device) 
                    for key, val in encoding.items()}
        outputs = self.text_model(**encoding)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation.
        projected = self.projection(cls_output)
        return projected

###########################################
# Custom Dataset
###########################################
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0, use_alt_text_encoder=True):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings  # used only if not using alt encoder
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0
        self.use_alt_text_encoder = use_alt_text_encoder

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        # Only used if not using alternative text encoder:
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

                # Split text into question and answer.
                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()
                label = 1 if raw_label == "match" else 0

                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.label_data.append(label)
                # Only load precomputed embeddings if not using alt text encoder.
                if not self.use_alt_text_encoder:
                    self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                    self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
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
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        img_name = os.path.basename(img_path)
        question_text = self.question_data[idx]
        # If using alternative text encoder, return raw strings.
        if self.use_alt_text_encoder:
            q_feature = question_text
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
# Image-Text Matching Model with Ensemble option and Joint/Hybrid Text Encoder
###########################################
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None, use_alt_text_encoder=False):
        print(f'BUILDING {ARCHITECTURE} model, pretrained={PRETRAINED}, use_alt_text_encoder={use_alt_text_encoder}')
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE
        self.use_alt_text_encoder = use_alt_text_encoder

        # For the vision branches, we keep our ensemble options.
        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
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
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)
        
        elif self.ARCHITECTURE == "Ensemble":
            self.cnn_model = models.resnet18(pretrained=PRETRAINED)
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
            self.vit_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vit_model.num_features, 128)
            # Fusion module for weighted fusion of CNN and ViT features.
            self.fusion = FusionModule(feature_dim=128)
        
        else:
            raise ValueError("UNKNOWN neural architecture")

        # For text, if not using alternative text encoder, use your projection layers.
        if not self.use_alt_text_encoder:
            self.question_embedding_layer = nn.Linear(768, 128)
            self.answer_embedding_layer = nn.Linear(768, 128)
        else:
            # Instantiate the alternative text encoder (shared for both question and answer)
            self.alt_text_encoder = AlternativeTextEncoder(model_name="distilbert-base-uncased", projection_dim=128)
        
        # Final classification layer.
        if self.ARCHITECTURE == "Ensemble":
            self.fc = nn.Linear(128 + 128 + 128, num_classes)
        else:
            self.fc = nn.Linear(128 + 128 + 128, num_classes)

    def forward(self, img, question_input, answer_input):
        # Process vision: determine feature extraction based on chosen architecture.
        if self.ARCHITECTURE == "CNN":
            img_features = self.vision_model(img)
        elif self.ARCHITECTURE == "ViT":
            img_features = self.vision_model(img)
            img_features = self.fc_vit(img_features)
        elif self.ARCHITECTURE == "Ensemble":
            cnn_features = self.cnn_model(img)
            vit_features = self.vit_model(img)
            vit_features = self.fc_vit(vit_features)
            img_features, gate_weights = self.fusion(cnn_features, vit_features)
        else:
            raise ValueError("Unsupported architecture")

        # Process text: if using alt text encoder, question_input and answer_input are strings.
        if self.use_alt_text_encoder:
            # DataLoader collates raw strings into a list.
            question_features = self.alt_text_encoder(question_input)
            answer_features = self.alt_text_encoder(answer_input)
        else:
            # Otherwise, they are precomputed tensor embeddings.
            question_features = self.question_embedding_layer(question_input)
            answer_features = self.answer_embedding_layer(answer_input)
        
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

###########################################
# Fusion Module: Weighted Fusion via gating
###########################################
class FusionModule(nn.Module):
    def __init__(self, feature_dim):
        super(FusionModule, self).__init__()
        self.fc = nn.Linear(2 * feature_dim, 2)
    def forward(self, cnn_feat, vit_feat):
        combined = torch.cat((cnn_feat, vit_feat), dim=1)
        weights = self.fc(combined)
        weights = torch.softmax(weights, dim=1)
        fused_feat = weights[:, 0].unsqueeze(1) * cnn_feat + weights[:, 1].unsqueeze(1) * vit_feat
        return fused_feat, weights

###########################################
# Training Function with timing and training accuracy
###########################################
def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING {ARCHITECTURE} model')
    model.train()
    overall_start = time.time()
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
            # For text inputs: if using alt text encoder, they are lists of strings; otherwise, tensors.
            if not USE_ALT_TEXT_ENCODER:
                question_input = question_input.to(device)
                answer_input = answer_input.to(device)
            outputs = model(images, question_input, answer_input)
            loss = criterion(outputs, labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        epoch_loss = running_loss / total_batches
        epoch_duration = time.time() - epoch_start

        # Evaluate training accuracy.
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
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
        train_accuracy = correct / total
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}] Average Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f} s, Training Accuracy: {train_accuracy:.4f}')
    overall_duration = time.time() - overall_start
    print(f'TOTAL Training Time: {overall_duration:.2f} seconds')

###########################################
# Evaluation Function: computes test loss, accuracy, and MRR (with timing)
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
    print(f'Testing Time: {eval_duration:.2f} seconds')

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
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.3, use_alt_text_encoder=USE_ALT_TEXT_ENCODER)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test", use_alt_text_encoder=USE_ALT_TEXT_ENCODER)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Choose architecture: "CNN", "ViT", or "Ensemble"
    MODEL_ARCHITECTURE = "Ensemble"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL, use_alt_text_encoder=USE_ALT_TEXT_ENCODER).to(device)
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
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}, USE_ALT_TEXT_ENCODER={USE_ALT_TEXT_ENCODER}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    
    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=30)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
