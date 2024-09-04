import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, cohen_kappa_score
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on a dataset')
    parser.add_argument("--exp", type=str, choices=['original', 'mixedhard'], default='original', help="experiment type. original = using original data trained classifier; mixedhard = using mixed hard data trained classifier")
    parser.add_argument("--model_dir", type=str, default=None, help="model directory")
    parser.add_argument("--output_dir", type=str, default=None, help="output directory")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_dir = args.model_dir
    # Load the BERT tokenizer and model.
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # Load the dataset.
    data = pd.read_json('/usa/dayu/CRS/T-CRS/A_comprehensive_exp/evaluator/annotation/redial_eval_dataset.json')
    print("data length: ", len(data))
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []

    for sent in data['text']:
        encoded_dict = tokenizer.encode_plus(
            sent,                      
            add_special_tokens = True, 
            max_length = 128,          
            pad_to_max_length = True,
            return_attention_mask = True,  
            return_tensors = 'pt', 
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(data['label'])

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=32)

    # Put model in evaluation mode
    model.eval()
    # model to device
    model.to(device)
    predictions, true_labels = [], []

    output_df = pd.DataFrame(columns=['Input Text Part 1', 'Input Text Part 2', 'Ground Truth Label', 'Model Prediction'])

    
    # Predict 
    for batch in tqdm(prediction_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
    
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        
        # Get the predicted labels for this batch
        batch_predictions = np.argmax(logits, axis=1)

        # Convert input_ids back to text
        batch_texts = [tokenizer.decode(input_id, skip_special_tokens=False) for input_id in b_input_ids]
        batch_texts_part1 = [text.split('[SEP]')[0].replace('[CLS]', '').strip() if '[SEP]' in text else text for text in batch_texts]
        batch_texts_part2 = [text.split('[SEP]')[1].strip() if '[SEP]' in text else '' for text in batch_texts]

        # Append results to the DataFrame
        batch_df = pd.DataFrame({
            'Input Text Part 1': batch_texts_part1,
            'Input Text Part 2': batch_texts_part2,
            'Ground Truth Label': label_ids,
            'Model Prediction': batch_predictions
        })
        output_df = pd.concat([output_df, batch_df], ignore_index=True)
    
        predictions.append(logits)
        true_labels.append(label_ids)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    output_df.to_csv(args.output_dir + f'redial_evaluation_results_{args.exp}.csv', index=False)
    
    print('Total Accuracy:', accuracy_score(flat_true_labels, flat_predictions))
    print("Cohen's Kappa:", cohen_kappa_score(flat_true_labels, flat_predictions))



if __name__ == '__main__':
    main()