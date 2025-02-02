import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model
model = BertForSequenceClassification.fropm_pretrained(
    "bert-base-uncased", num_labels=5)

# Save the model
save_path = "model/bert_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved at {save_path}")
