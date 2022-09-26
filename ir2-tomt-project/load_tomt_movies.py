import json
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# config = PretrainedConfig.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
# config.num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2').to(device)
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
# dummy data for testing
queries = ['What is New York famous for?'] * 32
docs = ['New York City is famous for the Metropolitan Museum of Art.'] * 32 

class TomtDataset(Dataset):

	def __init__(self, tokenizer, path=None):
		self.data = dict(tokenizer(queries,
		                           docs,
		                           truncation=True,
		                           padding='max_length',
		                           return_tensors='pt'))

		self.data['labels'] = torch.tensor([1] * self.data['input_ids'].shape[0],
									       dtype=torch.long) # Labels are size (B, ), one for each batch element.

	def __len__(self):
		return len( self.data['labels'] )

	def __getitem__(self, idx):
		return self.data['input_ids'][idx], self.data['token_type_ids'][idx], self.data['attention_mask'][idx], self.data['labels'][idx]

data = TomtDataset(tokenizer)
dataloader = DataLoader(data, batch_size=4)

for input_ids, token_type_ids, attention_mask, labels in dataloader:
	
	

	model_output = model(input_ids=input_ids.to(device),
	                     token_type_ids=token_type_ids.to(device),
	                     attention_mask=attention_mask.to(device),
	                     labels=labels.to(device))

	print(model_output.loss)
	print(model_output.logits)
	

	break