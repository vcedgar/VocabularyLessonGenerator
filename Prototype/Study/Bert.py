#referenced from: https://www.analyticsvidhya.com/blog/2019/07/pytorch-transformers-nlp-python/

import torch
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import pandas
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.fr import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES

lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize input
file = open('./Books/text.txt',mode='r')

tokenized_text = tokenizer.tokenize(file.read())

vocabulary = pandas.read_csv("./Lexique382/lexique3_words_90_percentile.tsv", sep='\t').lemme.to_list()
unknownVocab = {}
index = 0
for token in tokenized_text:
    lemma = lemmatizer.lookup(token)
    if lemma not in vocabulary and token.isalpha():
        unknownVocab[token] = index
        tokenized_text[index] = '[MASK]'
    index += 1

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])


# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')
model.eval()

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]

for token in unknownVocab.keys():
    predicted_index = torch.argmax(predictions[0, unknownVocab[token]]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, "...", token)