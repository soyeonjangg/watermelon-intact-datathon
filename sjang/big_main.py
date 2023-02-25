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
# target_list = [' Emergency Room Reports', ' Surgery', ' Radiology', ' Podiatry',
#        ' Neurology', ' Gastroenterology', ' Orthopedic',
#        ' Cardiovascular / Pulmonary', ' Nephrology',
#        ' ENT - Otolaryngology', ' General Medicine',
#        ' Hematology - Oncology', ' Cosmetic / Plastic Surgery',
#        ' SOAP / Chart / Progress Notes', ' Chiropractic',
#        ' Psychiatry / Psychology', ' Consult - History and Phy.',
#        ' Hospice - Palliative Care', ' Neurosurgery',
#        ' Obstetrics / Gynecology', ' Urology', ' Discharge Summary',
#        ' Autopsy', ' Dermatology', ' Letters', ' Office Notes',
#        ' Lab Medicine - Pathology', ' Ophthalmology',
#        ' Speech - Language', ' Dentistry', ' Pediatrics - Neonatal',
#        ' Physical Medicine - Rehab', ' Bariatrics', ' Endocrinology',
#        ' Pain Management', ' IME-QME-Work Comp etc.',
#        ' Allergy / Immunology', ' Sleep Medicine',
#        ' Diets and Nutritions', ' Rheumatology']

new_target_list = [
    0, 1, 2, 3, 4, 5, 6, 7
]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE =  1e-5

orig_train_df = pd.read_csv("./data/train.csv")

X, y = orig_train_df.transcription, orig_train_df.new_labels

# 여기부터 수정 예정
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=200)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)

train_medical_df = pd.get_dummies(train_df.new_labels)
val_medical_df = pd.get_dummies(val_df.new_labels)

train_df = pd.concat([train_df, train_medical_df], axis=1)
val_df = pd.concat([val_df, val_medical_df], axis=1)
# class_weights = class_weight.compute_class_weight(class_weight='balanced',  classes=np.unique(target_list), y=train_df.medical_specialty)
# weights= torch.tensor(class_weights,dtype=torch.float)

# train_size = 0.8
# train_df = orig_train_df.sample(frac=train_size, random_state=200)
# train_df.drop('medical_specialty', axis=1, inplace=True)
# val_df.drop('medical_specialty', axis=1, inplace=True)

# val_df = orig_train_df.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN, new_target_list)
val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN, new_target_list)

train_data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size =TRAIN_BATCH_SIZE, num_workers = 0)
val_data_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size =VALID_BATCH_SIZE, num_workers = 0)

# GPU usage
if torch.cuda.is_available():        
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CPU exists.')

# weights = weights.to(device)


model = BERTClass(len(new_target_list))
model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

ckpt_path = "data/multi-label/curr_ckpt_smote"
model_path = "data/multi-label/best_model_class_smote.pt"

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, ckpt_path, model_path, device)

best_model_path = 'data/multi-label/best_model1.pt'
best_checkpoint_path = "data/multi-label/curr_ckpt1"

model.load_state_dict(torch.load(model_path)['state_dict'])
model.eval()

TEST_BATCH_SIZE = 32
medical_specialty_df = pd.DataFrame(columns=new_target_list)
test_df = pd.read_csv("data/new_test.csv")
test_df = pd.concat([test_df, medical_specialty_df])
test_df.drop('Unnamed: 0', axis=1, inplace=True)
test_df = test_df.fillna(0)
test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, new_target_list)
test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size =TEST_BATCH_SIZE, num_workers = 0)

# inference 
pred = predict(model, test_data_loader, device)

prob = []
for l in pred:
    max_val = max(l)
    print(max_val)
    idx = l.index(max_val)
    print(idx)
    prob.append(idx)


# print(torch.cuda.memory_summary(device=None, abbreviated=False))
