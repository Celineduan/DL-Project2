# Import required libraries
import os
import pandas as pd
import torch
from transformers import RobertaModel, RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, RobertaForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from datasets import load_dataset, Dataset, ClassLabel
import pickle
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

# Load tokenizer and model
base_model = 'roberta-base'

dataset = load_dataset('ag_news', split='train')
tokenizer = RobertaTokenizer.from_pretrained(base_model)

def preprocess(examples):
    tokenized = tokenizer(examples['text'], truncation=True, padding=True)
    return tokenized

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# Extract the number of classes and their names
num_labels = dataset.features['label'].num_classes
class_names = dataset.features["label"].names
print(f"number of labels: {num_labels}")
print(f"the labels: {class_names}")

# Create an id2label mapping
id2label = {i: label for i, label in enumerate(class_names)}

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Load pre-trained model
model = RobertaForSequenceClassification.from_pretrained(
    base_model,
    id2label=id2label)

# Split the dataset
split_datasets = tokenized_dataset.train_test_split(test_size=640, seed=42, stratify_by_column="labels")
train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# Examine class distribution to ensure balance
label_counts = train_dataset['labels'].value_counts()
print(f"Label distribution in training set: {label_counts}")

# PEFT Config - Enhanced configuration for better performance
peft_config = LoraConfig(
    r=9,  # Chosen to stay under 1M parameter limit
    lora_alpha=32,  # Higher alpha for stronger adaptation
    lora_dropout=0.1,  # Added dropout for regularization
    bias="none",
    # Target attention matrices for comprehensive adaptation
    target_modules=["query", "key", "value"],
    task_type=TaskType.SEQ_CLS,  # Explicitly set the task type
)

# Create the PEFT model by applying LoRA to the base model
peft_model = get_peft_model(model, peft_config)

# Function to count trainable parameters to ensure we stay under 1 million
def count_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_params

# Check if we're under the parameter limit
trainable_params, all_params = count_trainable_parameters(peft_model)
print(f"Trainable parameters: {trainable_params:,}")
print(f"All parameters: {all_params:,}")
print(f"Under 1M parameter limit: {'Yes' if trainable_params < 1_000_000 else 'No'}")

# Print trainable parameters to understand what's being fine-tuned
print("Trainable parameters:")
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        print(name)

print('PEFT Model')
peft_model.print_trainable_parameters()

# To track evaluation metrics during training
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate accuracy and other metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Setup Training args with improved settings for better performance
output_dir = "results"
training_args = TrainingArguments(
    output_dir=output_dir,
    report_to=None,  # Disable reporting to save resources
    evaluation_strategy='steps',
    eval_steps=100,  # Evaluate more frequently 
    logging_steps=50,
    learning_rate=3e-4,  # Increased learning rate for LoRA
    num_train_epochs=3,  # Train for more epochs
    max_steps=-1,  # Set to -1 to use num_train_epochs instead of a fixed number of steps
    weight_decay=0.01,  # Add L2 regularization
    per_device_train_batch_size=32,  # Increased batch size
    per_device_eval_batch_size=64,
    fp16=torch.cuda.is_available(),  # Use mixed precision if available
    gradient_accumulation_steps=2,  # Accumulate gradients for stability
    warmup_ratio=0.1,  # Add warmup steps
    load_best_model_at_end=True,  # Save the best model
    metric_for_best_model="accuracy",
    greater_is_better=True,
    dataloader_num_workers=4,
    optim="adamw_torch",  # Use AdamW optimizer
)

def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

# Start Training
peft_lora_finetuning_trainer = get_trainer(peft_model)

# Train the model and capture results
result = peft_lora_finetuning_trainer.train()

# Display training results
print(f"Training completed with loss: {result.metrics['train_loss']:.4f}")
print(f"Training time: {result.metrics['train_runtime']:.2f} seconds")

# Inference function
def classify(model, tokenizer, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(**inputs)

    # Get prediction and convert to probabilities
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    prediction = logits.argmax(dim=-1).item()
    confidence = probabilities[0][prediction].item()

    print(f'\nClass: {prediction}, Label: {id2label[prediction]}, Confidence: {confidence:.4f}')
    print(f'Text: {text}')
    
    # Show all class probabilities
    print("\nProbabilities for all classes:")
    for i, label in id2label.items():
        print(f"{label}: {probabilities[0][i].item():.4f}")
        
    return id2label[prediction]

# Test with examples to verify model performance
classify(peft_model, tokenizer, "Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his innocence...")
classify(peft_model, tokenizer, "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.")
classify(peft_model, tokenizer, "Google announces new smartphone with advanced AI capabilities at annual developer conference")
classify(peft_model, tokenizer, "Scientists discover potential cure for cancer in rainforest plant, clinical trials to begin next year")

# Evaluation function
def evaluate_model(inference_model, dataset, labelled=True, batch_size=8, data_collator=None):
    """
    Evaluate a PEFT model on a dataset.
    """
    # Create the DataLoader
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_model.to(device)
    inference_model.eval()

    all_predictions = []
    if labelled:
        all_labels = []

    # Loop over the DataLoader
    for batch in tqdm(eval_dataloader):
        # Move each tensor in the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = inference_model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_predictions.append(predictions.cpu())
        
        if labelled:
            # Expecting that labels are provided under the "labels" key.
            references = batch["labels"]
            all_labels.append(references.cpu())

    # Concatenate predictions from all batches
    all_predictions = torch.cat(all_predictions, dim=0)

    if labelled:
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels.numpy(), all_predictions.numpy())
        precision = precision_score(all_labels.numpy(), all_predictions.numpy(), average='weighted')
        recall = recall_score(all_labels.numpy(), all_predictions.numpy(), average='weighted')
        f1 = f1_score(all_labels.numpy(), all_predictions.numpy(), average='weighted')
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels.numpy(), all_predictions.numpy())
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[id2label[i] for i in range(len(id2label))],
            yticklabels=[id2label[i] for i in range(len(id2label))]
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        print(f"Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }, all_predictions
    else:
        return all_predictions

# Check evaluation accuracy on validation set
eval_metrics, _ = evaluate_model(peft_model, eval_dataset, True, 32, data_collator)
print(f"\nValidation accuracy: {eval_metrics['accuracy']:.4f}")

# Process unlabelled test data
try:
    print("\nProcessing unlabelled test data...")
    unlabelled_dataset = pd.read_pickle("test_unlabelled.pkl")
    
    # For unlabelled dataset, check its structure
    print(f"Unlabelled dataset keys: {unlabelled_dataset.column_names if hasattr(unlabelled_dataset, 'column_names') else 'N/A'}")
    
    # If it's a pandas DataFrame, convert to HF Dataset
    if isinstance(unlabelled_dataset, pd.DataFrame):
        print("Converting DataFrame to Dataset")
        unlabelled_dataset = Dataset.from_pandas(unlabelled_dataset)
    
    # Preprocess the test data the same way as training data
    test_dataset = unlabelled_dataset.map(preprocess, batched=True, remove_columns=["text"])
    print(f"Loaded {len(test_dataset)} unlabelled test samples")

    # Run inference and save predictions
    print("Running inference on test dataset...")
    preds = evaluate_model(peft_model, test_dataset, False, 32, data_collator)
    
    df_output = pd.DataFrame({
        'ID': range(len(preds)),
        'Label': preds.numpy()  # Convert to numpy for saving
    })
    
    # Save predictions to CSV for submission
    submission_path = os.path.join(output_dir, "inference_output.csv")
    df_output.to_csv(submission_path, index=False)
    print(f"Inference complete. Predictions saved to {submission_path}")
    
    # Check the first few predictions
    print("\nSample predictions:")
    for i in range(min(5, len(df_output))):
        pred_class = df_output['Label'][i]
        print(f"Sample {i}: Predicted class: {pred_class} ({id2label[pred_class]})")
        
except Exception as e:
    print(f"Error processing unlabelled test data: {e}")
    print("Skipping test data processing - you'll need to rerun this section when you have the test data.")

# Final parameter check
final_trainable_params, final_all_params = count_trainable_parameters(peft_model)
print("\nFinal Model Parameters:")
print(f"Total parameters: {final_all_params:,}")
print(f"Trainable parameters: {final_trainable_params:,}")
print(f"Percentage of parameters that are trainable: {100 * final_trainable_params / final_all_params:.2f}%")
print(f"Under 1M parameter limit: {'Yes' if final_trainable_params <= 1_000_000 else 'No'}")

# Save the final model
final_model_path = os.path.join(output_dir, "final_model")
peft_lora_finetuning_trainer.save_model(final_model_path)
print(f"Final model saved to {final_model_path}")

print("\nTraining and evaluation complete! The model is ready for submission.")
print(f"Final accuracy: {eval_metrics['accuracy']:.4f}")
print(f"The model uses {final_trainable_params:,} trainable parameters (limit: 1,000,000)")