import torch
import pandas as pd



class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len, target_list):
        
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transcription = self.df['transcription']
        self.targets = self.df[target_list].values

    def __len__(self):
        return len(self.transcription)

    def __getitem__(self, index):
        transcription = str(self.transcription[index])
        transcription = " ".join(transcription.split())

        inputs = self.tokenizer.encode_plus(transcription, None, add_special_tokens=True, max_length=self.max_len, padding='max_length', return_token_type_ids=True, truncation=True, return_attention_mask=True, return_tensors='pt')
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }