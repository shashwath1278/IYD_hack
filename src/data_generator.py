"""
Enhanced data generator for Ollama Llama 3.8B fine-tuning
Creates comprehensive training datasets for Ramayana fact-checking
"""

import json
import jsonlines
import random
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path

class RamayanaDataGenerator:
    def __init__(self):
        self.ramayana_knowledge = self._load_extended_knowledge()
    
    def _load_extended_knowledge(self) -> Dict:
        """Extended Ramayana knowledge base for training data generation."""
        return {
            "characters": {
                "Rama": {
                    "titles": ["Prince of Ayodhya", "Maryada Purushottama", "Seventh Avatar of Vishnu"],
                    "family": {
                        "father": "Dasharatha",
                        "mother": "Kausalya",
                        "wife": "Sita",
                        "brothers": ["Lakshmana", "Bharata", "Shatrughna"]
                    },
                    "attributes": ["righteousness", "dharma", "archery skills", "divine nature"],
                    "weapons": ["bow", "arrows", "Brahmastra"]
                },
                "Sita": {
                    "titles": ["Princess of Mithila", "Daughter of Earth"],
                    "family": {
                        "father": "Janaka",
                        "husband": "Rama"
                    },
                    "birthplace": "Mithila",
                    "special_events": ["Swayamvara", "Agni Pariksha", "exile to forest"]
                },
                "Lakshmana": {
                    "relation": "Rama's younger brother",
                    "mother": "Sumitra",
                    "loyalty": "devoted to Rama",
                    "exile": "accompanied Rama and Sita"
                },
                "Ravana": {
                    "titles": ["King of Lanka", "Ten-headed demon"],
                    "heads": 10,
                    "kingdom": "Lanka",
                    "abilities": ["shapeshifting", "flying", "immense strength"],
                    "downfall": "pride and ego",
                    "defeated_by": "Rama"
                },
                "Hanuman": {
                    "titles": ["Son of Vayu", "Devotee of Rama", "Sankat Mochan"],
                    "father": "Vayu (Wind God)",
                    "abilities": ["flight", "size changing", "superhuman strength", "crossing oceans"],
                    "role": "messenger and devotee"
                }
            },
            "places": {
                "Ayodhya": {
                    "description": "Capital of Kosala kingdom",
                    "ruler": "Dasharatha",
                    "significance": "Rama's birthplace and kingdom"
                },
                "Lanka": {
                    "description": "Island kingdom of Ravana",
                    "location": "across the ocean from India",
                    "ruler": "Ravana"
                },
                "Mithila": {
                    "description": "Kingdom of Janaka",
                    "significance": "Sita's birthplace",
                    "ruler": "Janaka"
                },
                "Dandaka_Forest": {
                    "description": "Forest where the trio lived in exile",
                    "duration": "part of 14-year exile"
                }
            },
            "events": {
                "birth_of_rama": "Rama was born to Dasharatha and Kausalya",
                "sita_swayamvara": "Rama won Sita by breaking Shiva's bow",
                "exile": "Rama was exiled for 14 years due to Kaikeyi's boons",
                "golden_deer": "Maricha disguised as golden deer to lure Rama away",
                "sita_abduction": "Ravana abducted Sita while Rama was away",
                "hanuman_meets_rama": "Hanuman met Rama and became his devotee",
                "sundara_kanda": "Hanuman's journey to Lanka to find Sita",
                "bridge_construction": "Vanaras built bridge to Lanka under Nala and Nila",
                "lanka_war": "Great war between Rama's army and Ravana's forces",
                "ravana_death": "Ravana was killed by Rama's arrow",
                "agni_pariksha": "Sita underwent fire trial to prove her purity",
                "rama_coronation": "Rama was crowned king of Ayodhya"
            },
            "concepts": {
                "dharma": "righteousness and moral duty",
                "exile_duration": "14 years",
                "vanara_army": "army of monkeys who helped Rama",
                "rama_rajya": "ideal rule of Rama",
                "ramayana_author": "Sage Valmiki"
            }
        }
    
    def generate_true_statements(self, count: int = 50) -> List[str]:
        """Generate factually correct statements about Ramayana."""
        statements = []
        kb = self.ramayana_knowledge
        
        # Character-based facts
        statements.extend([
            "Rama is the son of Dasharatha and Kausalya",
            "Sita is the wife of Rama and daughter of Janaka",
            "Lakshmana is Rama's younger brother",
            "Ravana is the ten-headed king of Lanka",
            "Hanuman is the son of Vayu and devotee of Rama",
            "Bharata and Shatrughna are brothers of Rama",
            "Sumitra is the mother of Lakshmana",
            "Janaka is the father of Sita",
            "Dasharatha is the king of Ayodhya"
        ])
        
        # Event-based facts
        statements.extend([
            "Rama was exiled for 14 years",
            "Sita was abducted by Ravana",
            "Hanuman crossed the ocean to reach Lanka",
            "A bridge was built to Lanka by the vanaras",
            "Rama defeated Ravana in battle",
            "Sita underwent Agni Pariksha",
            "Rama won Sita's hand by breaking Shiva's bow",
            "The golden deer was actually Maricha in disguise"
        ])
        
        # Place-based facts
        statements.extend([
            "Ayodhya is the kingdom of Rama",
            "Lanka is the kingdom of Ravana",
            "Mithila is the birthplace of Sita",
            "The Dandaka forest was where they lived during exile"
        ])
        
        return statements[:count]
    
    def generate_false_statements(self, count: int = 50) -> List[str]:
        """Generate factually incorrect statements about Ramayana."""
        statements = [
            "Rama has ten heads",
            "Sita is the daughter of Ravana",
            "Hanuman is the king of Lanka",
            "Rama was exiled for 7 years",
            "Ravana is Rama's brother",
            "Lakshmana is the son of Ravana",
            "Sita was born in Lanka",
            "Dasharatha is the king of Lanka",
            "Hanuman is the father of Rama",
            "Ravana defeated Rama in battle",
            "Sita is the mother of Ravana",
            "Ayodhya is located in Lanka",
            "Rama has five heads",
            "Lakshmana abducted Sita",
            "Hanuman built the bridge to Lanka alone",
            "Ravana was exiled for 14 years",
            "Sita is the king of Mithila",
            "Rama is the son of Ravana",
            "Lanka is located in Ayodhya",
            "Hanuman has ten heads"
        ]
        
        return statements[:count]
    
    def generate_partially_true_statements(self, count: int = 20) -> List[str]:
        """Generate statements that are partially correct."""
        statements = [
            "Rama was exiled for 10 years",  # Should be 14
            "Ravana has five heads",  # Should be 10
            "Sita is the daughter of Dasharatha",  # Wrong father
            "Hanuman is the son of Rama",  # Wrong father
            "Lakshmana is the king of Ayodhya",  # Wrong person
            "Rama defeated Hanuman in battle",  # Wrong opponent
            "Sita was born in Ayodhya",  # Wrong place
            "Ravana is the prince of Lanka",  # Should be king
            "Hanuman crossed the river to reach Lanka",  # Should be ocean
            "Bharata was exiled with Rama"  # Bharata stayed back
        ]
        
        return statements[:count]
    
    def create_training_prompt(self, statement: str, verdict: str, confidence: float, evidence: str) -> Dict:
        """Create a training prompt in the format expected by Ollama."""
        
        context = json.dumps(self.ramayana_knowledge, indent=2)
        
        prompt = f"""You are an expert on the Ramayana, the ancient Indian epic. Your task is to fact-check claims about the Ramayana using the provided reference data.

REFERENCE DATA:
{context}

CLAIM TO VERIFY: "{statement}"

INSTRUCTIONS:
1. Analyze the claim against the reference data
2. Determine if the claim is TRUE, FALSE, or PARTIALLY_TRUE
3. Provide specific evidence from the reference data
4. If information is missing, state "INSUFFICIENT_DATA"

RESPONSE FORMAT:
VERDICT: [TRUE/FALSE/PARTIALLY_TRUE/INSUFFICIENT_DATA]
CONFIDENCE: [0.0-1.0]
EVIDENCE: [Specific references from the data]
EXPLANATION: [Brief explanation of your reasoning]

Please provide your fact-check analysis:"""

        response = f"""VERDICT: {verdict}
CONFIDENCE: {confidence}
EVIDENCE: {evidence}
EXPLANATION: {"This claim is supported by the canonical Ramayana narrative" if verdict == "TRUE" else "This claim contradicts established facts from the Ramayana" if verdict == "FALSE" else "This claim contains some accurate elements but also incorrect information"}"""

        return {
            "prompt": prompt,
            "response": response
        }
    
    def generate_ollama_training_dataset(self, output_file: str = "ramayana_ollama_training.jsonl"):
        """Generate comprehensive training dataset for Ollama fine-tuning."""
        
        training_data = []
        
        # Generate TRUE examples
        true_statements = self.generate_true_statements(40)
        for statement in true_statements:
            example = self.create_training_prompt(
                statement=statement,
                verdict="TRUE",
                confidence=0.95,
                evidence="Direct reference found in canonical Ramayana sources"
            )
            training_data.append(example)
        
        # Generate FALSE examples
        false_statements = self.generate_false_statements(40)
        for statement in false_statements:
            example = self.create_training_prompt(
                statement=statement,
                verdict="FALSE",
                confidence=0.90,
                evidence="Contradicts established facts from Ramayana"
            )
            training_data.append(example)
        
        # Generate PARTIALLY_TRUE examples
        partial_statements = self.generate_partially_true_statements(20)
        for statement in partial_statements:
            example = self.create_training_prompt(
                statement=statement,
                verdict="PARTIALLY_TRUE",
                confidence=0.75,
                evidence="Contains some accurate elements but also incorrect information"
            )
            training_data.append(example)
        
        # Shuffle the data
        random.shuffle(training_data)
        
        # Save to JSONL format
        with jsonlines.open(output_file, 'w') as writer:
            for item in training_data:
                writer.write(item)
        
        print(f"Generated {len(training_data)} training examples in {output_file}")
        
        # Also save as regular JSON for inspection
        json_file = output_file.replace('.jsonl', '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        return training_data

def main():
    """Generate training data for Ollama Llama 3.8B fine-tuning."""
    print("🎓 Generating Ollama Training Data for Ramayana Fact Checker")
    print("=" * 60)
    
    generator = RamayanaDataGenerator()
    
    # Generate training dataset
    training_data = generator.generate_ollama_training_dataset()
    
    print(f"\n✅ Generated {len(training_data)} training examples")
    print("\n📁 Files created:")
    print("  - ramayana_ollama_training.jsonl (for fine-tuning)")
    print("  - ramayana_ollama_training.json (for inspection)")
    
    print("\n🔧 Next steps:")
    print("1. Review the generated training data")
    print("2. Use the JSONL file for Ollama fine-tuning")
    print("3. Create custom model with Modelfile")

if __name__ == "__main__":
    main()
