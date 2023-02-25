import pandas as pd
import torch

from dataloader import CustomDataset
from model import BERTClass
from transformers import BertTokenizer
from train_utils import *

new_target_list = [
    0, 1, 2, 3, 4, 5, 6, 7
]

detail_labels = [
    [0, 10, 16, 17, 36], [14, 15, 28, 31, 35], [3, 4, 8, 22, 33],
    [1, 9, 18, 29, 34], [11, 13, 21, 30, 39], [6, 12, 19, 20, 27],
    [5, 7, 23, 32, 38], [2, 24, 25, 26, 37]
    ]

# GPU usage
if torch.cuda.is_available():        
    device = torch.device("cuda")
    print('GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('CPU exists.')


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 256
EPOCHS = 10
LEARNING_RATE =  1e-5

TEST_BATCH_SIZE = 32
medical_specialty_df = pd.DataFrame(columns=new_target_list)
test_df = pd.read_csv("data/new_test.csv")
test_df = pd.concat([test_df, medical_specialty_df])
test_df.drop('Unnamed: 0', axis=1, inplace=True)
test_df = test_df.fillna(0)
test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN, new_target_list)
test_data_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size =TEST_BATCH_SIZE, num_workers = 0)

psudo_label_path = "./data/best_model_class_smote0.pt"

model = BERTClass(len(new_target_list))
model.to(device)
import ipdb; ipdb.set_trace()

model.load_state_dict(torch.load(psudo_label_path)['state_dict'])
model.eval()
# inference 
pred = predict(model, test_data_loader, device)

prob = []
for l in pred:
    max_val = max(l)
    print(max_val)
    idx = l.index(max_val)
    print(idx)
    prob.append(idx)
