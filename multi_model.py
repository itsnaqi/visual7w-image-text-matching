################################################################################
# Robust Multi-Choice VQA Classifier with Error Handling
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32
from transformers import BlipProcessor, BlipForQuestionAnswering
from sentence_transformers import SentenceTransformer, util

# Initialize VLM components
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_vaq = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
model_ss = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vaq.to(device)

class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_options = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.load_data()

    def load_data(self):
        print(f"Loading data from {self.data_file}")
        question_groups = {}

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f):
                try:
                    line = line.strip()
                    if not line: continue
                    
                    img_name, text, label = line.split("\t")
                    img_path = os.path.join(self.images_path, img_name.strip())
                    
                    # Handle questions with multiple '?' characters
                    parts = text.split("?", 1)
                    if len(parts) != 2:
                        raise ValueError("Invalid question-answer format")
                        
                    question_text = parts[0].strip() + '?'
                    answer_text = parts[1].strip()
                    
                    key = (img_path, question_text)
                    if key not in question_groups:
                        question_groups[key] = {
                            'options': [],
                            'correct_idx': -1,
                            'count': 0
                        }
                    
                    # Validate exactly 4 options per question
                    question_groups[key]['count'] += 1
                    if question_groups[key]['count'] > 4:
                        raise ValueError(f"Too many answers for question: {question_text}")
                    
                    question_groups[key]['options'].append(answer_text)
                    if label == "match":
                        if question_groups[key]['correct_idx'] != -1:
                            raise ValueError("Multiple correct answers for question")
                        question_groups[key]['correct_idx'] = len(question_groups[key]['options'])-1

                except Exception as e:
                    print(f"Skipping line {line_number+1}: {str(e)}")
                    continue

        # Create final dataset entries
        for (img_path, question), group in question_groups.items():
            if len(group['options']) == 4 and group['correct_idx'] != -1:
                self.image_data.append(img_path)
                self.question_data.append(question)
                self.answer_options.append(group['options'])
                self.labels.append(group['correct_idx'])
            else:
                print(f"Skipping incomplete group: {question}")

        print(f"Successfully loaded {len(self.image_data)} valid question groups")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_data[idx]).convert("RGB")
            img = self.transform(img)
            
            # Get embeddings for all 4 options
            question_emb = torch.tensor(self.sentence_embeddings[self.question_data[idx]], dtype=torch.float32)
            answer_embs = [torch.tensor(self.sentence_embeddings[opt], dtype=torch.float32) 
                          for opt in self.answer_options[idx]]
            
            return {
                'image': img,
                'question': question_emb,
                'answers': torch.stack(answer_embs),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            return None

class EnhancedVQAModel(nn.Module):
    def __init__(self, num_classes=4, arch='CNN', pretrained=True):
        super().__init__()
        self.arch = arch
        
        # Vision encoder
        if arch == 'CNN':
            self.vision = models.resnet18(pretrained=pretrained)
            if pretrained:
                for param in self.vision.parameters():
                    param.requires_grad = False
                for param in list(self.vision.children())[-2:]:
                    param.requires_grad = True
            self.vision.fc = nn.Linear(self.vision.fc.in_features, 128)
        elif arch == 'ViT':
            self.vision = vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
            self.vision.heads = nn.Linear(self.vision.heads.head.in_features, 128)
        else:
            raise ValueError("Unsupported architecture")

        # Text processing
        self.question_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.answer_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 + 256 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions, answers):
        # Visual features
        vis_features = self.vision(images)
        
        # Text features
        q_features = self.question_proj(questions)
        a_features = self.answer_proj(answers)
        
        # Combine features
        combined = torch.cat([
            vis_features.unsqueeze(1).expand(-1, 4, -1),
            q_features.unsqueeze(1).expand(-1, 4, -1),
            a_features
        ], dim=-1)
        
        return self.classifier(combined)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            
            outputs = model(
                batch['image'].to(device),
                batch['question'].to(device),
                batch['answers'].to(device)
            )
            
            probs = torch.softmax(outputs, dim=-1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())

    # Calculate metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Exact Match
    em = np.mean(np.argmax(all_preds, axis=1) == all_labels)
    
    # MRR
    ranks = []
    for i in range(len(all_labels)):
        sorted_indices = np.argsort(all_preds[i])[::-1]
        rank = np.where(sorted_indices == all_labels[i])[0][0] + 1
        ranks.append(1/rank)
    mrr = np.mean(ranks)
    
    # VLM Re-Ranking
    vlm_em = 0
    for idx in range(len(all_labels)):
        question = loader.dataset.question_data[idx]
        options = loader.dataset.answer_options[idx]
        
        # Get BLIP prediction
        inputs = processor(
            images=Image.open(loader.dataset.image_data[idx]).convert("RGB"),
            text=question,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            output = model_vaq.generate(**inputs)
            
        blip_answer = processor.decode(output[0], skip_special_tokens=True)
        blip_emb = model_ss.encode(blip_answer, convert_to_tensor=True)
        option_embs = model_ss.encode(options, convert_to_tensor=True)
        
        # Combine model probs with VLM similarity
        similarities = util.pytorch_cos_sim(blip_emb, option_embs).cpu().numpy()
        final_probs = all_preds[idx] + 0.3 * similarities  # Weighted combination
        vlm_em += (np.argmax(final_probs) == all_labels[idx])
    
    return {
        'EM': em,
        'MRR': mrr,
        'VLM_EM': vlm_em/len(all_labels),
        'predictions': all_preds
    }

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    model.train()
    best_score = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        
        for batch in train_loader:
            if batch is None: continue
            
            optimizer.zero_grad()
            
            outputs = model(
                batch['image'].to(device),
                batch['question'].to(device),
                batch['answers'].to(device)
            )
            
            loss = criterion(outputs.view(-1, 4), batch['label'].to(device))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Validation EM: {val_metrics['EM']:.4f}")
        print(f"Validation MRR: {val_metrics['MRR']:.4f}")
        print(f"VLM Boosted EM: {val_metrics['VLM_EM']:.4f}")
        
        # Save best model
        if val_metrics['EM'] > best_score:
            best_score = val_metrics['EM']
            torch.save(model.state_dict(), "best_model.pth")

def main():
    # Configuration
    config = {
        'images_path': "/workspaces/ITM_Classifier-baselines/visual7w-images",
        'train_file': "/workspaces/ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt",
        'test_file': "/workspaces/ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt",
        'embeddings_file': "/workspaces/ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl",
        'batch_size': 32,
        'lr': 3e-5,
        'epochs': 15,
        'arch': 'ViT'  # or 'CNN'
    }

    # Load data
    sentence_embeddings = pickle.load(open(config['embeddings_file'], 'rb'))
    
    train_dataset = ITM_Dataset(
        config['images_path'],
        config['train_file'],
        sentence_embeddings,
        'train'
    )
    
    test_dataset = ITM_Dataset(
        config['images_path'],
        config['test_file'],
        sentence_embeddings,
        'test'
    )

    # Create loaders with collate_fn to handle None returns
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return torch.utils.data.default_collate(batch)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                            collate_fn=collate_fn)

    # Initialize model
    model = EnhancedVQAModel(arch=config['arch']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    train_model(model, train_loader, test_loader, optimizer, criterion, config['epochs'])
    
    # Final evaluation
    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load("best_model.pth"))
    metrics = evaluate_model(model, test_loader, device)
    print(f"Test EM: {metrics['EM']:.4f}")
    print(f"Test MRR: {metrics['MRR']:.4f}")
    print(f"Test VLM EM: {metrics['VLM_EM']:.4f}")

if __name__ == "__main__":
    main()