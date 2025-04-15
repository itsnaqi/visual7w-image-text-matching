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