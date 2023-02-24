import pandas as pd


def assign_new_label(label):
    if label in [0, 10, 16, 17, 36]:
        return 0
    elif label in [14, 15, 28, 31, 35]:
        return 1
    elif label in [3, 4, 8, 22, 33]:
        return 2
    elif label in [1, 9, 18, 29, 34]:
        return 3
    elif label in [11, 13, 21, 30, 39]:
        return 4
    elif label in [6, 12, 19, 20, 27]:
        return 5
    elif label in [5, 7, 23, 32, 38]:
        return 6
    elif label in [2, 24, 25, 26, 37]:
        return 7

def assign_label(url, save):
    df = pd.read_csv(url)
    df['new_labels'] = df['labels'].apply(assign_new_label)
    df = df.drop('Unnamed: 0', axis=1)
    df.to_csv(save,index=False)

assign_label("./data/new_train.csv", './data/train.csv')

