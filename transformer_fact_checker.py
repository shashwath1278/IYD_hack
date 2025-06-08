"""
Ramayana Fact Checker using DiffLlama Architecture
Open Source Solution using Hugging Face Transformers
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    Trainer, TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch.nn as nn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffLlamaFactChecker(nn.Module):
    """
    Advanced Fact Checker using DiffLlama-inspired architecture
    Implements differential attention mechanisms for better fact verification
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", num_labels: int = 3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load base model configuration
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        
        # Base transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model
        hidden_size = self.transformer.config.hidden_size
        
        # Differential attention layers inspired by DiffLlama
        self.lambda_init = 0.8  # Initial lambda value for differential attention
        self.attention_heads = 8
        self.head_dim = hidden_size // self.attention_heads
        
        # Multi-head differential attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size) 
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Lambda parameters for differential attention
        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        
        # Group normalization for differential attention
        self.groupnorm = nn.LayerNorm(hidden_size)
        
        # Fact verification specific layers
        self.fact_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=self.attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Pre-classifier with dropout
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using DiffLlama-style initialization"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize lambda parameters
        nn.init.normal_(self.lambda_q1, 0, 0.02)
        nn.init.normal_(self.lambda_k1, 0, 0.02)
        nn.init.normal_(self.lambda_q2, 0, 0.02)
        nn.init.normal_(self.lambda_k2, 0, 0.02)
    
    def differential_attention(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Implement differential attention mechanism inspired by DiffLlama
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.attention_heads, self.head_dim).transpose(1, 2)
        
        # Standard attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Split value states for differential attention
        v1, v2 = torch.chunk(v, 2, dim=1)
        v1 = v1.repeat(1, 2, 1, 1)
        v2 = v2.repeat(1, 2, 1, 1)
        
        # Apply attention to both value sets
        attn_output1 = torch.matmul(attn_weights, v1)
        attn_output2 = torch.matmul(attn_weights, v2)
        
        # Compute lambda values for differential combination
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).to(q.dtype)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).to(q.dtype)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        # Differential combination
        attn_output = attn_output1 - lambda_full * attn_output2
        attn_output = (1 - self.lambda_init) * self.groupnorm(attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with differential attention for fact verification"""
        
        # Get base transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply differential attention
        diff_attention_output = self.differential_attention(hidden_states)
        
        # Apply fact-specific attention
        fact_attended, _ = self.fact_attention(
            diff_attention_output, 
            diff_attention_output, 
            diff_attention_output,
            key_padding_mask=(attention_mask == 0) if attention_mask is not None else None
        )
        
        # Pool the sequence (use CLS token or mean pooling)
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(fact_attended.size()).float()
            pooled_output = torch.sum(fact_attended * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        else:
            # Use CLS token (first token)
            pooled_output = fact_attended[:, 0]
        
        # Apply pre-classifier
        pooled_output = self.pre_classifier(pooled_output)
        
        # Final classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'attention_output': fact_attended
        }

class RamayanaFactCheckerTrainer:
    """
    Advanced trainer for Ramayana Fact Checker using DiffLlama architecture
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None
        self.label_map = {"TRUE": 0, "FALSE": 1, "IRRELEVANT": 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
    def prepare_dataset(self, texts: List[str], labels: List[str]) -> Dataset:
        """Prepare dataset with advanced tokenization"""
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Convert labels to integers
        label_ids = [self.label_map[label] for label in labels]
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': label_ids
        })
        
        return dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(
                labels, predictions, 
                target_names=list(self.label_map.keys()),
                output_dict=True
            )
        }
    
    def train(self, train_texts: List[str], train_labels: List[str], 
              val_texts: List[str], val_labels: List[str],
              output_dir: str = "./ramayana_diffllama_model",
              num_epochs: int = 3,
              learning_rate: float = 2e-5,
              batch_size: int = 8):
        """Train the model with advanced techniques"""
        
        logger.info("Initializing DiffLlama-based Fact Checker...")
        
        # Initialize model
        self.model = DiffLlamaFactChecker(self.model_name)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        val_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            learning_rate=learning_rate,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("Starting DiffLlama training...")
        
        # Train the model
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        logger.info(f"Training completed!")
        logger.info(f"Final evaluation results: {eval_result}")
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mapping
        with open(f"{output_dir}/label_map.json", "w") as f:
            json.dump(self.label_map, f)
        
        logger.info(f"Model saved to {output_dir}")
        
        return train_result, eval_result
    
    def predict(self, texts: List[str], model_path: str = None) -> List[str]:
        """Make predictions using the trained model"""
        
        if model_path and self.model is None:
            # Load model
            self.model = DiffLlamaFactChecker(self.model_name)
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu'))
            self.model.eval()
        
        # Tokenize inputs
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs['logits'], dim=-1)
        
        # Convert to labels
        predicted_labels = [self.reverse_label_map[pred.item()] for pred in predictions]
        
        return predicted_labels

def train_new_model():
    """Train a new model from scratch."""
    print("Training new Ramayana Fact Checker model...")
    print("=" * 50)
    
    # Initialize trainer
    trainer = RamayanaFactCheckerTrainer()
    
    # Create enhanced training data
    print("Creating enhanced training data...")
    texts, labels = trainer.create_enhanced_training_data()
    
    print(f"Total training examples: {len(texts)}")
    print(f"Label distribution:")
    for label_name, label_id in trainer.label_map.items():
        count = labels.count(label_id)
        print(f"  {label_name}: {count}")
    
    # Train model
    print("\nTraining model (this may take 10-30 minutes)...")
    trainer.train_model(texts, labels)
    
    print("\n✓ Model training completed!")
    print("Model saved to './ramayana_model'")

def test_trained_model():
    """Test the trained model."""
    print("Testing trained model...")
    print("=" * 50)
    
    # Load fact checker
    checker = RamayanaFactCheckerTransformer()
    
    if not checker.classifier:
        print("✗ Model not found. Please train the model first.")
        return
    
    # Test statements
    test_statements = [
        "Rama was the son of King Dasharatha",
        "Sita was found in a furrow when Janaka was plowing", 
        "Hanuman was Ravana's brother",
        "Rama had four brothers",
        "Lakshmana accompanied Rama to the forest",
        "Ravana had ten heads",
        "What is machine learning?",
        "Python is a programming language",
        "Bharata ruled Ayodhya while Rama was in exile"
    ]
    
    print("Test Results:")
    print("-" * 30)
    
    for statement in test_statements:
        result = checker.check_fact(statement)
        detailed = checker.get_detailed_prediction(statement)
        
        if result is True:
            status = "✓ TRUE"
        elif result is False:
            status = "✗ FALSE"
        else:
            status = "? IRRELEVANT"
        
        confidence = detailed['top_prediction']['confidence']
        print(f"Statement: {statement}")
        print(f"Result: {status} (confidence: {confidence:.3f})")
        print("-" * 30)

def interactive_mode():
    """Interactive testing mode."""
    print("Ramayana Fact Checker - Interactive Mode")
    print("Using Fine-tuned DistilBERT Model")
    print("=" * 50)
    
    checker = RamayanaFactCheckerTransformer()
    
    if not checker.classifier:
        print("✗ Model not found. Please train the model first.")
        print("Run: python transformer_fact_checker.py --train")
        return
    
    print("Enter statements to check, or 'quit' to exit.")
    print("Type 'detail' before a statement for detailed analysis.")
    
    while True:
        user_input = input("\nEnter statement: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Check for detailed mode
        if user_input.lower().startswith('detail '):
            statement = user_input[7:]
            detailed = checker.get_detailed_prediction(statement)
            print(f"\nDetailed Analysis:")
            print(f"Statement: {detailed['statement']}")
            print("Predictions:")
            for pred in detailed['predictions']:
                print(f"  {pred['label']}: {pred['confidence']:.3f}")
        else:
            result = checker.check_fact(user_input)
            if result is True:
                print("✓ FACTUALLY CORRECT")
            elif result is False:
                print("✗ FACTUALLY INCORRECT") 
            else:
                print("? NOT RELEVANT TO RAMAYANA")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_new_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_trained_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        print("Ramayana Fact Checker - Transformer Edition")
        print("=" * 50)
        print("Usage:")
        print("  python transformer_fact_checker.py --train      # Train new model")
        print("  python transformer_fact_checker.py --test       # Test trained model")
        print("  python transformer_fact_checker.py --interactive # Interactive mode")
        print()
        print("First time setup:")
        print("1. Install requirements: pip install torch transformers datasets scikit-learn")
        print("2. Train model: python transformer_fact_checker.py --train")
        print("3. Test model: python transformer_fact_checker.py --test")
