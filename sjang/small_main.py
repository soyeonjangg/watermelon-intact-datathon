from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torchsummary import summary
from sklearn.model_selection import train_test_split
from dataloader import CustomDataset
from model import BERTClass
from ckpoint import *
from train_utils import predict, flat_accuracy, train_model

# GPU usage
if torch.cuda.is_available():        
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CPU exists.')


detail_labels = [
    [0, 10, 16, 17, 36], [14, 15, 28, 31, 35], [3, 4, 8, 22, 33],
    [1, 9, 18, 29, 34], [11, 13, 21, 30, 39], [6, 12, 19, 20, 27],
    [5, 7, 23, 32, 38], [2, 24, 25, 26, 37]
    ]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE =  1e-5

orig_train_df = pd.read_csv("./data/train.csv")

for i, label in enumerate(detail_labels):

    df = orig_train_df[orig_train_df['new_labels'] == i]
    X, y = df.transcription, df.labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=200)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    train_medical_df = pd.get_dummies(train_df.labels)
    val_medical_df = pd.get_dummies(val_df.labels)

    train_df = pd.concat([train_df, train_medical_df], axis=1)
    val_df = pd.concat([val_df, val_medical_df], axis=1)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, label)
    val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN, label)
    
    train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size =TRAIN_BATCH_SIZE, num_workers = 0)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size =VALID_BATCH_SIZE, num_workers = 0)
    
    model = BERTClass(len(label))
    model.to(device)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    ckpt_path = "data/curr_ckpt_smote"+ str(i) + ".pt"
    model_path = "data/best_model_class_smote"+ str(i) + ".pt"

    trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, model_path, device)

