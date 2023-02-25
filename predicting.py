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

pseudo_label_path = "./data/best_model_class.pt"
#######
model = BERTClass(len(new_target_list))
model.to(device)

model.load_state_dict(torch.load(pseudo_label_path)['state_dict'])
model.eval()
# inference 
preds = predict(model, test_data_loader, device)

pseudo_labels = []
for pred in preds:
    pseudo_labels.append(pred.index(max(pred)))
test_df = test_df[['transcription']]
test_df['pseudo_labels'] = pseudo_labels

final_result = pd.DataFrame(columns=['index', 'transcription', 'Predicted'])
########
for i, label in enumerate(detail_labels):
    real_label_path = "./data/best_model_class" + str(i) + ".pt"
    tmp_df = test_df[test_df['pseudo_labels'] == i]
    tmp_df.reset_index(inplace=True, drop=False)
    
    medical_specialty_df = pd.DataFrame(columns=label)
    tmp_df = pd.concat([tmp_df, medical_specialty_df])
    tmp_df = tmp_df.fillna(0)

    tmp_dataset = CustomDataset(tmp_df, tokenizer, MAX_LEN, label)
    tmp_dataloader = torch.utils.data.DataLoader(tmp_dataset, shuffle=False, batch_size =TEST_BATCH_SIZE, num_workers = 0)

    model = BERTClass(len(label))
    model.to(device)

    model.load_state_dict(torch.load(real_label_path)['state_dict'])
    model.eval()
    preds = predict(model, tmp_dataloader, device)

    real_labels = []
    for pred in preds:
        pos = pred.index(max(pred))
        real_labels.append(label[pos])
    
    tmp_df['Predicted'] = real_labels
    final_result = final_result.append(tmp_df[['index', 'transcription', 'Predicted']], ignore_index = True)

try:
    final_result.sort_values(by=['index'])
    final_result.drop(['index', 'transcription'], axis=1, inplace=True)
    final_result.reset_index(drop=True, inplace=True)

    final_result.to_csv('./data/final.csv', index=True)
except:
    import ipdb; ipdb.set_trace()

import pandas as pd
result = pd.read_csv("./data/final.csv")
result.Predicted.value_counts()