Resnet50
EVALUATING CNN model
Balanced Accuracy: 0.6504, 32.18 seconds
Total Test Loss: 168.5419

EVALUATING ViT model DeiT (Data-efficient Image Transformer)
Balanced Accuracy: 0.5857, 33.21 seconds
Total Test Loss: 386.4507

Epoch [30/30] Average Loss: 0.3998, Duration: 65.65 s, Training Accuracy: 0.8210
TOTAL Training Time: 3613.52 seconds
EVALUATING Ensemble model
Test Accuracy: 0.8020
Balanced Accuracy: 0.8020
Total Test Loss: 0.4618
Mean Reciprocal Rank (MRR): 0.7832
Testing Time: 27.22 seconds

Epoch [30/30] Average Loss: 0.4143, Duration: 42.35 s, Training Accuracy: 0.8185
TOTAL Training Time: 2765.58 seconds
EVALUATING CNN model
Test Accuracy: 0.8048
Balanced Accuracy: 0.8048
Total Test Loss: 0.4448
Mean Reciprocal Rank (MRR): 0.7823
Testing Time: 16.10 seconds

Epoch [30/30] Average Loss: 0.3892, Duration: 72.50s, Training Accuracy: 0.8331
TOTAL Training Time: 3974.37 seconds
EVALUATING Ensemble model FusionModule
Test Accuracy: 0.7968
Balanced Accuracy: 0.7968
Total Test Loss: 0.4666
Mean Reciprocal Rank (MRR): 0.7829
Testing Time: 25.17 seconds

Epoch [30/30] Average Loss: 0.4054, Duration: 65.73 s, Training Accuracy: 0.8220
TOTAL Training Time: 3815.72 seconds
EVALUATING ViT model
Test Accuracy: 0.7982
Balanced Accuracy: 0.7982
Total Test Loss: 0.4559
Mean Reciprocal Rank (MRR): 0.7816
Testing Time: 26.76 seconds

Epoch [30/30] Average Loss: 0.1490, Duration: 95.90 s, Training Accuracy: 0.9511
TOTAL Training Time: 4845.99 seconds
EVALUATING Ensemble model
Test Accuracy: 0.7759
Balanced Accuracy: 0.7759
Total Test Loss: 0.9780
Mean Reciprocal Rank (MRR): 0.7757
Testing Time: 27.70 seconds


Alternative Text Encoder Module using DistilBERT with Dropout  # Attention-based Fusion Module using Multihead Attention
Epoch [7/15] Average Loss: 0.0202, Duration: 129.46 s, Training Accuracy: 0.9262, Validation Accuracy: 0.8044
Early stopping triggered.
TOTAL Training Time: 1565.53 seconds
EVALUATING Ensemble model
Test Accuracy: 0.7978
Balanced Accuracy: 0.7978
Total Test Loss: 0.0716
Mean Reciprocal Rank (MRR): 0.7820
Precision: 0.6603
Recall: 0.3940
F1-Score: 0.4935
AUC: 0.7744
Testing Time: 35.87 seconds
Explanation of Key Changes:

    Regularization with Dropout:

        In AlternativeTextEncoder and Transformer_VisionEncoder, dropout layers (with a rate adjustable via dropout_rate) are applied to the text and vision features respectively.

        An additional dropout (self.dropout in the CNN branch of ITM_Model) is applied after the CNN features.

    Early Stopping:

        The train_model() function now takes a dev_loader (validation set) and a patience parameter.

        After each epoch, the validation accuracy is computed using evaluate_accuracy().

        Training stops early if validation accuracy does not improve for patience consecutive epochs. The best model state is retained.

    Class Imbalance and Loss Function:

        A custom FocalLoss class is provided. Set USE_FOCAL_LOSS=True to use it (with default gamma=2 and alpha weight of 0.75).

        Otherwise, standard CrossEntropyLoss is used.

    Learning Rate Scheduler and Parameter Groups:

        In get_optimizer_and_scheduler(), a separate (smaller) learning rate (text_lr) is assigned for the text encoder parameters when USE_ALT_TEXT_ENCODER is True.

    Attention-Based Fusion Module:

        AttentionFusionModule uses multi-head attention (with 2 heads by default) to combine the CNN and ViT features in a more expressive manner.

    Additional Metrics:

        The evaluation function now calculates precision, recall, F1-score, and AUC (area under the ROC curve) along with accuracy, test loss, and Mean Reciprocal Rank (MRR).

    ViT Change:

        The vision encoder now uses vit_base_patch16_224 from timm (which is a ViT-B/16 model) instead of vit_b_32. This offers a finer patch resolution.


**Your ensemble model with the RoBERTa text encoder**
Epoch [7/15] Loss: 0.0159, Duration: 144.08s, Train Acc: 0.9480, Val Acc: 0.8059 
Early stopping triggered.
TOTAL Training Time: 1831.11 seconds
EVALUATING Ensemble model
Test Accuracy: 0.8012
Balanced Accuracy: 0.8012
Total Test Loss: 0.0832
Mean Reciprocal Rank (MRR): 0.7865
Precision: 0.6700
Recall: 0.4033
F1-Score: 0.5035
AUC: 0.7644
Testing Time: 40.14 seconds

Epoch [7/15] Loss: 0.0181, Duration: 143.80s, Train Acc: 0.9376, Val Acc: 0.8021
Early stopping triggered.
TOTAL Training Time: 1691.93 seconds
EVALUATING Ensemble model
Test Accuracy: 0.7923
Balanced Accuracy: 0.7923
Total Test Loss: 0.0758
Mean Reciprocal Rank (MRR): 0.7770
Precision: 0.6353
Recall: 0.3973
F1-Score: 0.4889
AUC: 0.7677
Testing Time: 36.44 seconds
Model saved to trained_model.pth

Epoch [8/15] Loss: 0.1567, Duration: 145.70s, Train Acc: 0.9479, Val Acc: 0.7936  roberta focal off
Early stopping triggered.
TOTAL Training Time: 1923.17 seconds
EVALUATING Ensemble model
Test Accuracy: 0.7860
Balanced Accuracy: 0.7860
Total Test Loss: 0.7103
Mean Reciprocal Rank (MRR): 0.7792
Precision: 0.5810
Recall: 0.5157
F1-Score: 0.5464
AUC: 0.7695
Testing Time: 39.23 seconds
Model saved to trained_model.pth

TRAINING model
Epoch [1/15] Train Loss: 0.5696, Val Loss: 0.5595, Train Acc: 0.7482, Val Acc: 0.7500
Epoch [2/15] Train Loss: 0.5584, Val Loss: 0.5389, Train Acc: 0.7482, Val Acc: 0.7500
Epoch [3/15] Train Loss: 0.5391, Val Loss: 0.5142, Train Acc: 0.7724, Val Acc: 0.7743
Epoch [4/15] Train Loss: 0.5230, Val Loss: 0.4991, Train Acc: 0.7873, Val Acc: 0.7889
Epoch [5/15] Train Loss: 0.5114, Val Loss: 0.4906, Train Acc: 0.7912, Val Acc: 0.7909
Epoch [6/15] Train Loss: 0.5052, Val Loss: 0.4895, Train Acc: 0.7917, Val Acc: 0.7946
Epoch [7/15] Train Loss: 0.5019, Val Loss: 0.4873, Train Acc: 0.7923, Val Acc: 0.7949
Epoch [8/15] Train Loss: 0.4973, Val Loss: 0.4821, Train Acc: 0.7948, Val Acc: 0.7951
Epoch [9/15] Train Loss: 0.4934, Val Loss: 0.4822, Train Acc: 0.7950, Val Acc: 0.7963
Epoch [10/15] Train Loss: 0.4960, Val Loss: 0.4823, Train Acc: 0.7949, Val Acc: 0.7959
Epoch [11/15] Train Loss: 0.4950, Val Loss: 0.4813, Train Acc: 0.7950, Val Acc: 0.7968
Epoch [12/15] Train Loss: 0.4947, Val Loss: 0.4809, Train Acc: 0.7961, Val Acc: 0.7968
Epoch [13/15] Train Loss: 0.4961, Val Loss: 0.4803, Train Acc: 0.7962, Val Acc: 0.7969
Epoch [14/15] Train Loss: 0.4935, Val Loss: 0.4782, Train Acc: 0.7972, Val Acc: 0.7963
Epoch [15/15] Train Loss: 0.4950, Val Loss: 0.4781, Train Acc: 0.7970, Val Acc: 0.7988
Precision: 0.7729, Recall: 0.2368, F1: 0.3625, AUC: 0.7326
Model saved to trained_model.pth