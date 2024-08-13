from unittest.util import strclass
# from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset


# #### SET VALUES FOR THESE ####

# seed = 42   ## IMPORTANT: CHANGE THIS FOR THE THREE EXPERIMENT RUNS [42, 1000, 2000]
# dataset = 'reddit'   #["gab", "twitter", "reddit", "youtube", "implicit"]
# num_epochs = 3

### END SETTING VALUES ###

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## set dataset here
    # options = ["gab", "twitter", "reddit", "youtube"]


    # dataset = "youtube"

    # file_map = {
    #     "gab": '/content/Rationales_file_GAB_dataset_corrected.csv',
    #     "twitter": '/content/Rationales_file_TWITTER_dataset.csv',
    #     "reddit": '/content/Rationales_file_REDDIT_dataset.csv',
    #     "youtube": '/content/Rationales_file_YOUTUBE_dataset.csv'
    # }

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

    model_name = 'GroNLP/HateBERT'
    tokenizer = BertTokenizer.from_pretrained("GroNLP/hateBERT")
    model = BertForSequenceClassification.from_pretrained("GroNLP/hateBERT")

    class CustomDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            texts = self.texts[idx]
            labels = self.labels[idx]
            encoding = self.tokenizer(texts, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
            # input_ids, mask_ids = torch.tensor(encoding['input_ids']), torch.tensor(encoding['attention_mask'])
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()
            labels = labels
            return input_ids, attention_mask, labels

    # Hyperparameters for tuning model initially. Let's see, we will change if required.
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    #Splitting training and validation testing split to test accuracy
    if args.dataset=='implicit':
        train_text, val_texts, train_labels, val_labels = train_test_split(train_df['post'].tolist(),train_df['label'].tolist(), test_size = 0.2)
    else:
        train_text, val_texts, train_labels, val_labels = train_test_split(train_df['text'].tolist(),train_df['label'].tolist(), test_size = 0.2)
    train_dataset = CustomDataset(train_text, train_labels, tokenizer, max_length = 512)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length = 512)

    #Creating dataloader object to train the model
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    model = model.to(device)

    from tqdm import tqdm

    # num_epochs = 3
    for epoch in range(args.num_epochs):
        model.train()

        train_losses = []
        train_accuracy = 0
        train_epoch_size = 0

        with tqdm(train_dataloader, desc=f'Epoch {epoch + 1}', dynamic_ncols=True) as loop:
            for batch in loop:
                input_ids, mask_ids, labels = batch
                if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    labels = labels.to(device)
                # optimizer.grad()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=mask_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Update accuracy and epoch size
                predictions = torch.argmax(outputs.logits, dim=1)
                train_accuracy += (predictions == labels).sum().item()
                train_epoch_size += len(labels)

                # Update tqdm progress bar with set_postfix
                # loop.set_postfix(loss=loss.item(), accuracy=train_accuracy / train_epoch_size)


        # Evaluating on Validation task
        model.eval()

        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, mask_ids, labels = batch
                if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    labels = labels.to(device)
                outputs = model(input_ids=input_ids, attention_mask=mask_ids)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_predictions, val_labels)
        print(f"Epoch {epoch + 1}: Validation Accuracy: {accuracy:.4f}")

    # torch.save(model, f'fine_tuned_naive_hatebert_{dataset}_{seed}.pt')

    if args.dataset=='implicit':
        test_texts = test_df['post'].tolist()
    else:
        test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    test_dataset = CustomDataset(test_texts, test_labels, tokenizer, max_length = 512)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)

    model.eval()
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids, mask_ids, labels = batch
            if torch.cuda.is_available():
                    input_ids = input_ids.to(device)
                    mask_ids = mask_ids.to(device)
                    labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=mask_ids)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(test_predictions, test_labels)

    print(f"Dataset: {args.dataset}, Seed: {args.seed}, Epochs: {args.num_epochs}")
    print("Accuracy of test dataset:", accuracy)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--seed', type=str, default=42)
    parser.add_argument('--dataset', type=str, default='gab')

    args = parser.parse_args()

    main(args)