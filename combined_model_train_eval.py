from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW

import torch
import argparse
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    file_map = {
            "gab": '/scratch/abhatt43/HSData/Rationales_file_GAB_dataset_corrected.csv',
            "twitter": '/scratch/abhatt43/HSData/Rationales_file_TWITTER_dataset.csv',
            "reddit": '/scratch/abhatt43/HSData/Rationales_file_REDDIT_dataset.csv',
            "youtube": '/scratch/abhatt43/HSData/Rationales_file_YOUTUBE_dataset.csv',
            "implicit": '/scratch/abhatt43/HSData/Rationales_file_IMPLICIT_hatespeech_dataset.csv'
        }

    file_path = file_map[args.dataset]
    df = pd.read_csv(file_path)
    train_df = df[df['exp_split'] == 'train']
    test_df = df[df['exp_split'] == 'test']

    print("Train df: ", len(train_df))
    print("Test_df: ", len(test_df))

    import gc
    # del variables
    gc.collect()

    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT") ## need this for tokenizing the input text in data loader
    tokenizer_bert = AutoTokenizer.from_pretrained(bert_model_name)

    class AdditionalCustomDataset(Dataset):
        def __init__(self, texts, labels, additional_texts, tokenizer, bert_tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.additional_texts = additional_texts
            self.tokenizer = tokenizer
            self.bert_tokenizer = bert_tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            texts = self.texts[idx]
            additional_texts = self.additional_texts[idx]
            labels = self.labels[idx]
            encoding = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            additional_encoding = self.bert_tokenizer(additional_texts, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            original_input_ids = encoding['input_ids'].squeeze()
            additional_input_ids = additional_encoding['input_ids'].squeeze()
            input_ids = torch.cat((encoding["input_ids"], additional_encoding["input_ids"]), dim=1)
            original_attention_mask = encoding['attention_mask'].squeeze()
            additional_attention_mask = additional_encoding['attention_mask'].squeeze()
            attention_mask = torch.cat((encoding["attention_mask"], additional_encoding["attention_mask"]), dim=1)
            labels = labels
            return original_input_ids, original_attention_mask, additional_input_ids, additional_attention_mask, labels
            # return input_ids, attention_mask, labels
            # return encoding, additional_encoding, labels

    #Splitting training and validation testing split to test accuracy
    if args.dataset=='implicit':
        train_text, val_texts, train_labels, val_labels = train_test_split(train_df['post'].tolist(),train_df['label'].tolist(), test_size = 0.2)
    else:
        train_text, val_texts, train_labels, val_labels = train_test_split(train_df['text'].tolist(),train_df['label'].tolist(), test_size = 0.2)
    
    add_train_text, add_val_texts, add_train_labels, add_val_labels = train_test_split(train_df['ChatGPT_Rationales'].tolist(),train_df['label'].tolist(), test_size = 0.2)


    ## Creating a CustomDataset
    train_dataset = AdditionalCustomDataset(train_text, train_labels, add_train_text, tokenizer, tokenizer_bert, max_length = 512)
    val_dataset = AdditionalCustomDataset(val_texts, val_labels, add_val_texts, tokenizer, tokenizer_bert, max_length = 512)

    #Creating dataloader object to train the model
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    class ProjectionMLP(nn.Module):
        def __init__(self, input_size, output_size):
            super(ProjectionMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, 2)
            )

        def forward(self, x):
            return self.layers(x)


    class ConcatModel(nn.Module):
        def __init__(self, hatebert_model, additional_model, projection_mlp, freeze_additional_model=True):
            super(ConcatModel, self).__init__()
            self.hatebert_model = hatebert_model
            self.additional_model = additional_model
            self.projection_mlp = projection_mlp

            if freeze_additional_model:
                for param in self.additional_model.parameters():
                    param.requires_grad = False


        def forward(self, input_ids, attention_mask, additional_input_ids, additional_attention_mask):
            # Forward pass through the HateBERT model
            hatebert_outputs = self.hatebert_model(input_ids=input_ids, attention_mask=attention_mask)
            hatebert_embeddings = hatebert_outputs.last_hidden_state[:, 0, :]  # Assuming [CLS] token representation
            hatebert_embeddings = torch.nn.LayerNorm(hatebert_embeddings.size()[1:]).to(device)(hatebert_embeddings.to(device)).to(device)
            # hatebert_embeddings = hatebert_embeddings.to(device)


            # Forward pass through the Additional Model
            additional_outputs = self.additional_model(input_ids=additional_input_ids, attention_mask=additional_attention_mask)
            additional_embeddings = additional_outputs.last_hidden_state[:, 0, :]  # Assuming [CLS] token representation

            additional_embeddings = torch.nn.LayerNorm(additional_embeddings.size()[1:]).to(device)(additional_embeddings.to(device)).to(device)

            # Concatenate the embeddings
            concatenated_embeddings = torch.cat((hatebert_embeddings, additional_embeddings), dim=1).to(device)
            # print("Size of concatenated embeddings:", concatenated_embeddings.size())

            # Project concatenated embeddings
            projected_embeddings = self.projection_mlp(concatenated_embeddings).to(device)

            return projected_embeddings



    hatebert_model = BertModel.from_pretrained("GroNLP/HateBERT").to(device)
    additional_model = BertModel.from_pretrained("bert-base-uncased").to(device)
    projection_mlp = ProjectionMLP(input_size=1536, output_size=512).to(device)

    if args.freeze=='yes':
        concat_model = ConcatModel(hatebert_model=hatebert_model, additional_model=additional_model, projection_mlp=projection_mlp, freeze_additional_model=True)
    elif args.freeze=='no':
        concat_model = ConcatModel(hatebert_model=hatebert_model, additional_model=additional_model, projection_mlp=projection_mlp, freeze_additional_model=False)
    
    concat_model = concat_model.to(device)

    optimizer = AdamW(concat_model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss().to(device)

    # criterion = criterion.to(device)

    from tqdm import tqdm

    for epoch in range(args.num_epochs):
        concat_model.train()

        train_losses = []
        train_accuracy = 0
        train_epoch_size = 0

        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', dynamic_ncols=True) as loop:
            for batch in loop:
                input_ids, attention_mask, additional_input_ids, additional_attention_mask, labels = batch

                if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    additional_input_ids = additional_input_ids.to(device)
                    additional_attention_mask = additional_attention_mask.to(device)
                    labels = labels.to(device)

                # Forward pass through the ConcatModel
                optimizer.zero_grad()
                outputs = concat_model(input_ids=input_ids, attention_mask=attention_mask, additional_input_ids=additional_input_ids, additional_attention_mask=additional_attention_mask)
                loss = criterion(outputs, labels)
                loss = criterion(outputs.view(-1, 2), labels.view(-1)) # 2 is number of labels

                # #Added Regularization -- To reduce overfitting
                # l2_lambda = 0.01
                # l2_reg = torch.tensor(0.).to(device)
                # for param in concat_model.parameters():
                #     l2_reg += torch.norm(param)
                # loss += l2_lambda * l2_reg

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Update accuracy and epoch size
                predictions = torch.argmax(outputs, dim=1)
                train_accuracy += (predictions == labels).sum().item()
                train_epoch_size += len(labels)

                # Update tqdm progress bar with set_postfix
                # loop.set_postfix(loss=loss.item(), accuracy=train_accuracy / train_epoch_size)

        # Evaluation on the validation set
        concat_model.eval()

        val_predictions = []
        val_labels = []

        with torch.no_grad(), tqdm(val_dataloader, desc='Validation', dynamic_ncols=True) as loop:
            for batch in loop:
                input_ids, attention_mask, additional_input_ids, additional_attention_mask, labels = batch

                if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    additional_input_ids = additional_input_ids.to(device)
                    additional_attention_mask = additional_attention_mask.to(device)
                    labels = labels.to(device)

                # Forward pass through the ConcatModel
                outputs = concat_model(input_ids=input_ids, attention_mask=attention_mask, additional_input_ids=additional_input_ids, additional_attention_mask=additional_attention_mask)
                sm = nn.Softmax(dim=1)
                predictions = torch.argmax(sm(outputs), dim=1)
                # print("prediction: ", predictions)
                # sm = nn.Softmax(dim=1)
                # predictions2 = sm(outputs)
                # print("prediction2: ", predictions2)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate and print validation accuracy
        accuracy = accuracy_score(val_predictions, val_labels)
        print(f"Epoch {epoch + 1}: Validation Accuracy: {accuracy:.4f}, Avg. Train Loss: {sum(train_losses) / len(train_losses):.4f}")

    if args.dataset=='implicit':
        test_texts = test_df['post'].tolist()
    else:
        test_texts = test_df['text'].tolist()

    add_test_texts = test_df['ChatGPT_Rationales'].tolist()
    test_labels = test_df['label'].tolist()

    test_dataset = AdditionalCustomDataset(test_texts, test_labels, add_test_texts, tokenizer, tokenizer_bert, max_length = 512)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    concat_model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, attention_mask, additional_input_ids, additional_attention_mask, labels = batch
            if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    additional_input_ids = additional_input_ids.to(device)
                    additional_attention_mask = additional_attention_mask.to(device)
                    labels = labels.to(device)
            outputs = concat_model(input_ids=input_ids, attention_mask=attention_mask, additional_input_ids=additional_input_ids, additional_attention_mask=additional_attention_mask)
            predictions = torch.argmax(outputs, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(test_predictions, test_labels)

    print(f"Dataset: {args.dataset}, Seed: {args.seed}, BERT frozen: {args.freeze}, Epochs: {args.num_epochs}")
    print("Accuracy of test dataset:", accuracy)


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--seed', type=str, default=42)
    parser.add_argument('--dataset', type=str, default='gab')
    parser.add_argument('--freeze', type=str, choices=['yes','no']) # whether to freeze additional bert model

    args = parser.parse_args()

    main(args)
