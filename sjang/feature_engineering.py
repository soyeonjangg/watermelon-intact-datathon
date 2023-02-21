import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# import nltk

train_df = pd.read_csv("data/new_train.csv")
train_df.head()

# lowercase
train_df['preprocessed_transcription'] = train_df['transcription'].str.lower()

# remove puncutations
def remove_punctuations(text):
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(text)
    return lst
train_df['preprocessed_transcription'] = train_df['preprocessed_transcription'].apply(lambda x: remove_punctuations(x))

# remove stop words
stop = stopwords.words('english')
train_df['preprocessed_transcription'] = train_df['preprocessed_transcription'].apply(lambda x: [item for item in x if item not in stop])

# lemmatize
wl = WordNetLemmatizer()
def lemmatizer(text):
    # verb
    # https://stackoverflow.com/questions/32957895/wordnetlemmatizer-not-returning-the-right-lemma-unless-pos-is-explicit-python
    # may have to use pos_tag
    lemm_text = [wl.lemmatize(word, 'v') for word in text]
    return lemm_text

train_df['preprocessed_transcription'] = train_df['preprocessed_transcription'].apply(lambda x:lemmatizer(x))

# stem
# Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.
ps = PorterStemmer()
train_df['preprocessed_transcription'] = train_df['preprocessed_transcription'].apply(lambda x: [ps.stem(y) for y in x])
train_df.to_csv('preprocessed_train_df.csv')

################################
# test data###
##
test_df = pd.read_csv("data/new_test.csv")
test_df.head()

# lowercase
test_df['preprocessed_transcription'] = train_df['transcription'].str.lower()

# remove puncutations
def remove_punctuations(text):
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(text)
    return lst
test_df['preprocessed_transcription'] = test_df['preprocessed_transcription'].apply(lambda x: remove_punctuations(x))

# remove stop words
stop = stopwords.words('english')
test_df['preprocessed_transcription'] = test_df['preprocessed_transcription'].apply(lambda x: [item for item in x if item not in stop])

# lemmatize
wl = WordNetLemmatizer()
def lemmatizer(text):
    # verb
    # https://stackoverflow.com/questions/32957895/wordnetlemmatizer-not-returning-the-right-lemma-unless-pos-is-explicit-python
    # may have to use pos_tag
    lemm_text = [wl.lemmatize(word, 'v') for word in text]
    return lemm_text

test_df['preprocessed_transcription'] = test_df['preprocessed_transcription'].apply(lambda x:lemmatizer(x))

# stem
# Stemming is the process of reducing inflection in words to their root forms such as mapping a group of words to the same stem even if the stem itself is not a valid word in the Language.
ps = PorterStemmer()
test_df['preprocessed_transcription'] = test_df['preprocessed_transcription'].apply(lambda x: [ps.stem(y) for y in x])
test_df.to_csv('preprocessed_test_df.csv')
