import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from fact_checker import RamayanaFactChecker

class RamayanaTrainingDataGenerator:
    def __init__(self, text_files_dir: str):
        """
        Generate training data for Ramayana fact-checking.
        
        Args:
            text_files_dir: Directory containing Ramayana text files
        """
        self.text_files_dir = Path(text_files_dir)
        self.ramayana_text = self._load_ramayana_texts()
        
        # Character mappings for generating false statements
        self.character_mappings = {
            'rama': ['ravana', 'hanuman', 'lakshmana', 'bharata'],
            'sita': ['kaikeyi', 'kausalya', 'sumitra', 'shabari'],
            'ravana': ['rama', 'hanuman', 'sugriva', 'vali'],
            'hanuman': ['ravana', 'rama', 'angada', 'jambavan'],
            'lakshmana': ['bharata', 'shatrughna', 'rama', 'ravana'],
            'dasharatha': ['janaka', 'sugriva', 'ravana', 'vali'],
            'ayodhya': ['lanka', 'mithila', 'kishkindha', 'panchavati'],
            'lanka': ['ayodhya', 'mithila', 'kishkindha', 'dandaka']
        }
        
        self.irrelevant_statements = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "The sun rises in the east",
            "Water boils at 100 degrees Celsius",
            "Shakespeare wrote Hamlet",
            "Einstein developed the theory of relativity",
            "The Great Wall of China is very long",
            "Football is played with eleven players",
            "The internet connects computers worldwide",
            "Democracy is a form of government"
        ]
    
    def _load_ramayana_texts(self) -> str:
        """Load and combine all Ramayana text files."""
        combined_text = ""
        
        for file_path in self.text_files_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    combined_text += f"\n\n{content}"
                    print(f"Loaded: {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return combined_text
    
    def _extract_factual_statements(self) -> List[str]:
        """Extract factual statements from the text."""
        statements = []
        
        # Simple patterns to extract statements
        patterns = [
            r'([A-Z][a-zA-Z\s]+) was the (\w+) of ([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+) had (\w+) (\w+)',
            r'([A-Z][a-zA-Z\s]+) went to ([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+) killed ([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+) married ([A-Z][a-zA-Z\s]+)',
            r'([A-Z][a-zA-Z\s]+) ruled ([A-Z][a-zA-Z\s]+)',
        ]
        
        sentences = re.split(r'[.!?]+', self.ramayana_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:
                # Check if it contains Ramayana characters
                if any(char in sentence.lower() for char in ['rama', 'sita', 'hanuman', 'ravana', 'lakshmana']):
                    statements.append(sentence)
        
        # Add some manually crafted true statements
        manual_true_statements = [
            "Rama was the son of King Dasharatha",
            "Sita was found in a furrow when Janaka was plowing",
            "Lakshmana accompanied Rama to the forest",
            "Ravana had ten heads",
            "Hanuman was devoted to Rama",
            "Bharata ruled Ayodhya while Rama was in exile",
            "Dasharatha had three wives",
            "Rama spent fourteen years in exile",
            "Sita was abducted by Ravana",
            "Hanuman crossed the ocean to reach Lanka",
            "Jatayu tried to save Sita",
            "Sugriva was the king of monkeys",
            "Vali was Sugriva's brother",
            "Angada was Vali's son",
            "Kumbhakarna was Ravana's brother",
            "Indrajit was Ravana's son",
            "Rama killed Ravana in battle",
            "Vibhishana helped Rama",
            "The bridge to Lanka was built by monkeys",
            "Rama returned to Ayodhya after defeating Ravana"
        ]
        
        statements.extend(manual_true_statements)
        
        return list(set(statements))[:100]  # Remove duplicates and limit
    
    def _generate_false_statements(self, true_statements: List[str]) -> List[str]:
        """Generate false statements by modifying true ones."""
        false_statements = []
        
        for statement in true_statements:
            # Try different ways to make it false
            modified = statement.lower()
            
            # Method 1: Swap characters
            for original, replacements in self.character_mappings.items():
                if original in modified:
                    replacement = random.choice(replacements)
                    false_statement = statement.replace(original.title(), replacement.title())
                    false_statement = statement.replace(original, replacement)
                    if false_statement != statement:
                        false_statements.append(false_statement)
                        break
        
        # Add some manually crafted false statements
        manual_false_statements = [
            "Hanuman was Ravana's brother",
            "Rama had four brothers",
            "Sita was the daughter of Ravana",
            "Lakshmana was older than Rama",
            "Dasharatha ruled Lanka",
            "Ravana was a devotee of Rama",
            "Hanuman could not fly",
            "Bharata went to exile with Rama",
            "Sugriva was Ravana's ally",
            "Jatayu helped Ravana abduct Sita",
            "Kumbhakarna was always awake",
            "Indrajit was Hanuman's son",
            "Rama killed Hanuman in battle",
            "Vibhishana was loyal to Ravana till the end",
            "The bridge to Lanka was built by demons",
            "Rama never returned to Ayodhya"
        ]
        
        false_statements.extend(manual_false_statements)
        
        return list(set(false_statements))[:80]  # Remove duplicates and limit
    
    def generate_training_data(self) -> List[Dict]:
        """Generate complete training dataset."""
        print("Extracting factual statements...")
        true_statements = self._extract_factual_statements()
        
        print("Generating false statements...")
        false_statements = self._generate_false_statements(true_statements)
        
        training_data = []
        
        # Add true statements
        for statement in true_statements:
            training_data.append({
                "statement": statement,
                "label": "TRUE",
                "explanation": "This statement is factually correct according to Valmiki's Ramayana."
            })
        
        # Add false statements
        for statement in false_statements:
            training_data.append({
                "statement": statement,
                "label": "FALSE",
                "explanation": "This statement contradicts the facts in Valmiki's Ramayana."
            })
        
        # Add irrelevant statements
        for statement in self.irrelevant_statements:
            training_data.append({
                "statement": statement,
                "label": "IRRELEVANT",
                "explanation": "This statement is not related to the Ramayana."
            })
        
        # Shuffle the data
        random.shuffle(training_data)
        
        print(f"Generated {len(training_data)} training examples:")
        print(f"- True statements: {len(true_statements)}")
        print(f"- False statements: {len(false_statements)}")
        print(f"- Irrelevant statements: {len(self.irrelevant_statements)}")
        
        return training_data
    
    def save_training_data(self, training_data: List[Dict], filename: str = "ramayana_training_data.jsonl"):
        """Save training data in JSONL format."""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"Training data saved to {filename}")
    
    def create_ollama_modelfile(self, training_data: List[Dict], filename: str = "RamayanaModelfile"):
        """Create an Ollama Modelfile for custom model creation."""
        
        # Create examples for the system prompt
        examples = []
        for item in training_data[:20]:  # Use first 20 examples
            examples.append(f"Statement: {item['statement']}\nLabel: {item['label']}")
        
        examples_text = "\n\n".join(examples)
        
        modelfile_content = f'''FROM llama3.1:8b

SYSTEM """You are a specialized Ramayana fact-checker based on Valmiki's Ramayana. Your task is to determine if statements about the Ramayana are factually correct.

INSTRUCTIONS:
- Respond with only one word: TRUE, FALSE, or IRRELEVANT
- TRUE: Statement is factually correct according to Valmiki's Ramayana
- FALSE: Statement contradicts the facts in Valmiki's Ramayana  
- IRRELEVANT: Statement is not related to the Ramayana

EXAMPLES:
{examples_text}

Remember: Consider synonyms and different name variations (e.g., Seetha/Sita, Lakshmana/Lakshman).
"""

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 10
'''

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"Ollama Modelfile saved to {filename}")
        print(f"To create the custom model, run: ollama create ramayana-checker -f {filename}")

def create_enhanced_training_data():
    """Create enhanced training data for the fact checker."""
    # Generate comprehensive training data
    true_statements = [
        "Rama was the son of King Dasharatha",
        "Sita was found in a furrow when Janaka was plowing",
        "Lakshmana accompanied Rama to the forest",
        "Ravana had ten heads",
        "Hanuman was devoted to Rama",
        "Bharata ruled Ayodhya while Rama was in exile",
        "Dasharatha had three wives",
        "Rama spent fourteen years in exile",
        "Sita was abducted by Ravana",
        "Hanuman crossed the ocean to reach Lanka"
    ]
    
    false_statements = [
        "Hanuman was Ravana's brother",
        "Rama had four brothers",
        "Sita was the daughter of Ravana",
        "Lakshmana was older than Rama",
        "Dasharatha ruled Lanka",
        "Ravana was a devotee of Rama",
        "Hanuman could not fly",
        "Bharata went to exile with Rama"
    ]
    
    irrelevant_statements = [
        "Python is a programming language",
        "Machine learning uses algorithms",
        "The sun rises in the east",
        "Water boils at 100 degrees Celsius",
        "Shakespeare wrote Hamlet"
    ]
    
    # Create training and validation sets
    train_data = []
    val_data = []
    
    # Add true statements
    for i, stmt in enumerate(true_statements):
        data_point = {"text": stmt, "label": "TRUE"}
        if i % 5 == 0:  # Every 5th item goes to validation
            val_data.append(data_point)
        else:
            train_data.append(data_point)
    
    # Add false statements
    for i, stmt in enumerate(false_statements):
        data_point = {"text": stmt, "label": "FALSE"}
        if i % 5 == 0:
            val_data.append(data_point)
        else:
            train_data.append(data_point)
    
    # Add irrelevant statements
    for i, stmt in enumerate(irrelevant_statements):
        data_point = {"text": stmt, "label": "IRRELEVANT"}
        if i % 5 == 0:
            val_data.append(data_point)
        else:
            train_data.append(data_point)
    
    return {
        "train": train_data,
        "validation": val_data
    }

def main():
    """Generate training data and create Ollama model."""
    
    generator = RamayanaTrainingDataGenerator("./data")
    
    print("Generating Ramayana training data...")
    training_data = generator.generate_training_data()
    
    print("\nSaving training data...")
    generator.save_training_data(training_data)
    
    print("\nCreating Ollama Modelfile...")
    generator.create_ollama_modelfile(training_data)
    
    print("\nNext steps:")
    print("1. Wait for llama3.1:8b to finish downloading")
    print("2. Run: ollama create ramayana-checker -f RamayanaModelfile")
    print("3. Update fact_checker.py to use model_name='ramayana-checker'")

if __name__ == "__main__":
    main()
