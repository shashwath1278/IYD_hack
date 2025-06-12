#!/usr/bin/env python3
"""
Enhanced Ramayana Statement Classifier with RAG Integration
Combines sophisticated RAG retrieval with Mistral model for accurate classification.
Author: Enhanced AI System
Version: 2.0 - Colab Optimized
"""

import pandas as pd
import torch
import os
import warnings
import subprocess
import sys
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
import time
from tqdm.auto import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# NLTK setup with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading required NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    _english_stopwords_set = set(stopwords.words('english'))
except ImportError:
    logger.warning("NLTK not available. Using basic text processing.")
    _english_stopwords_set = set()
    def sent_tokenize(text): return text.split('.')

# RAG Configuration
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 300
MAX_RETRIEVED_CHUNKS = 8
TOP_N_KEYWORDS_FOR_RETRIEVAL = 15

# Ramayana Knowledge Base
RAMAYANA_ENTITIES = {
    'rama': ['rama', 'raghava', 'dasharathi', 'kosala raja', 'ramachandra', 'maryada purushottam'],
    'sita': ['sita', 'seetha', 'janaki', 'vaidehi', 'maithili', 'bhoomija', 'earth daughter'],
    'ravana': ['ravana', 'dashanan', 'lankesh', 'rakshasa raja', 'ten headed', 'dashagriva'],
    'hanuman': ['hanuman', 'anjaneya', 'maruti', 'pavan putra', 'bajrangbali', 'wind son'],
    'lakshmana': ['lakshmana', 'lakshman', 'saumitri', 'rama brother'],
    'bharata': ['bharata', 'bharat', 'kaikeyi son'],
    'shatrughna': ['shatrughna', 'shatrughan', 'sumitra son'],
    'dasaratha': ['dasaratha', 'dasharath', 'ayodhya raja', 'ayodhya king'],
    'janaka': ['janaka', 'videha raja', 'mithila raja', 'sita father'],
    'sugriva': ['sugriva', 'sugreeva', 'monkey king', 'vali brother'],
    'vali': ['vali', 'bali', 'monkey king', 'sugriva brother'],
    'vibhishana': ['vibhishana', 'vibhishan', 'ravana brother', 'righteous demon'],
    'jatayu': ['jatayu', 'eagle', 'vulture king', 'bird king', 'great bird'],
    'kumbhakarna': ['kumbhakarna', 'sleeping giant', 'ravana brother']
}

CORE_RAMAYANA_FACTS = [
    'hanuman leap', 'hanuman ocean', 'hanuman lanka',
    'rama bridge', 'bridge ocean', 'setu',
    'ashoka vatika', 'sita garden', 'sita ravana',
    'ravana ten heads', 'ravana twenty arms',
    'rama ikshvaku', 'rama dynasty',
    'bharata sandals', 'bharata throne',
    'monkey army', 'bears army',
    'sanjeevani mountain', 'hanuman mountain'
]

def install_requirements():
    """Install required packages"""
    requirements = ["torch", "transformers", "accelerate", "sentencepiece", "protobuf"]
    for req in requirements:
        try:
            __import__(req.replace('-', '_'))
        except ImportError:
            logger.info(f"Installing {req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])

class RamayanaRAGSystem:
    """Sophisticated RAG system for Ramayana knowledge retrieval"""
    
    def __init__(self):
        self.chunks = []
        self.word_to_chunks = defaultdict(set)
        self.entity_to_chunks = defaultdict(set)
        self.reference_data = {}
        
    def load_reference_data(self):
        """Load Ramayana reference data from files with fallback knowledge"""
        # Try multiple possible data directories
        possible_dirs = ["data/", "./data/", "../data/", ""]
        data_files = [
            "valmiki_ramayan_supplementary_knowledge.txt",
            "valmiki_ramayan_bala_kanda_book1.txt",
            "valmiki_ramayan_ayodhya_kanda_book2.txt",
            "valmiki_ramayan_aranya_kanda_book3.txt",
            "valmiki_ramayan_kishkindha_kanda_book4.txt",
            "valmiki_ramayan_sundara_kanda_book5.txt",
            "valmiki_ramayan_yuddha_kanda_book6.txt"
        ]
        
        all_chunks = []
        files_found = 0
        
        for base_dir in possible_dirs:
            for filename in data_files:
                file_path = os.path.join(base_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content.strip()) > 100:
                            file_chunks = self._adaptive_chunking(content, file_path)
                            all_chunks.extend(file_chunks)
                            files_found += 1
                            logger.info(f"Loaded {len(file_chunks)} chunks from {file_path}")
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        # If no files found, create fallback knowledge base
        if not all_chunks:
            logger.warning("No Ramayana text files found. Creating fallback knowledge base...")
            all_chunks = self._create_fallback_knowledge()
        
        self.chunks = all_chunks
        self._build_index()
        logger.info(f"Total chunks loaded: {len(self.chunks)} from {files_found} files")
    
    def _create_fallback_knowledge(self) -> List[Dict]:
        """Create fallback knowledge base with essential Ramayana facts"""
        fallback_facts = [
            "Rama was the eldest son of King Dasharatha of Ayodhya. He was born to Queen Kaushalya and is considered the seventh avatar of Lord Vishnu.",
            
            "Sita was the daughter of King Janaka of Mithila. She was found in a furrow while plowing the earth, hence called Bhoomija (daughter of earth).",
            
            "Hanuman was the son of Vayu (wind god) and Anjana. He possessed immense strength and the ability to fly. He is known as the greatest devotee of Rama.",
            
            "Ravana was the ten-headed demon king of Lanka. He was learned in scriptures but was arrogant and lustful. He kidnapped Sita which led to the great war.",
            
            "Lakshmana was Rama's younger brother and constant companion. He was the son of Queen Sumitra and accompanied Rama during his 14-year exile.",
            
            "Bharata was another son of Dasharatha, born to Queen Kaikeyi. When Rama was exiled, Bharata refused to become king and ruled as regent with Rama's sandals on the throne.",
            
            "Rama spent 14 years in exile in the forest as per his father's promise to Kaikeyi. During this time, Ravana kidnapped Sita from their hermitage in Panchavati.",
            
            "Hanuman leaped across the ocean to reach Lanka in search of Sita. He found her in Ashoka Vatika, Ravana's garden, and gave her Rama's ring.",
            
            "Sugriva was the king of Kishkindha after his brother Vali was killed by Rama. He helped Rama by sending his monkey army to search for Sita.",
            
            "The war between Rama and Ravana lasted for several days. Rama killed Ravana with a divine arrow, and Sita was rescued.",
            
            "Vibhishana was Ravana's younger brother who joined Rama's side because he believed in dharma. He became the king of Lanka after Ravana's death.",
            
            "Jatayu was a great eagle who tried to save Sita when Ravana was carrying her away. He fought valiantly but was mortally wounded by Ravana.",
            
            "Rama returned to Ayodhya after 14 years and was crowned king. His rule, known as Rama Rajya, is considered the ideal of good governance.",
            
            "Kumbhakarna was Ravana's brother who slept for six months at a time. He was a mighty warrior who fought for Ravana despite knowing he was wrong.",
            
            "The bridge (Rama Setu) was built across the ocean to Lanka by the monkey army led by Nala and Nila, with Hanuman's help.",
            
            "Sanjeevani was the life-giving herb that Hanuman brought from the Dronagiri mountain to save Lakshmana when he was wounded by Indrajit's arrow.",
            
            "Shatrughna was the youngest of the four brothers, twin to Lakshmana. He stayed in Ayodhya to protect the kingdom while Rama was in exile.",
            
            "Urmila was Lakshmana's wife who sacrificed her marital life by staying in Ayodhya while Lakshmana served Rama in exile.",
            
            "Angada was the son of Vali and played a crucial role as Rama's messenger to Ravana before the war began.",
            
            "Indrajit (Meghnad) was Ravana's son and a powerful warrior who could become invisible. He was eventually killed by Lakshmana."
        ]
        
        chunks = []
        for i, fact in enumerate(fallback_facts):
            entities = self._extract_entities(fact)
            chunks.append({
                "text": fact,
                "source": f"fallback_knowledge_{i+1}",
                "entities": entities
            })
        
        return chunks
        
    def _adaptive_chunking(self, text: str, source_file: str) -> List[Dict]:
        """Create semantic chunks from text"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_entities = set()
        
        for sentence in sentences:
            sentence_entities = self._extract_entities(sentence)
            
            if len(current_chunk) + len(sentence) > CHUNK_SIZE and current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "source": os.path.basename(source_file),
                    "entities": list(current_entities)
                })
                
                # Smart overlap based on entities
                current_chunk = sentence
                current_entities = set(sentence_entities)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_entities.update(sentence_entities)
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": os.path.basename(source_file),
                "entities": list(current_entities)
            })
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract Ramayana entities from text"""
        text_lower = text.lower()
        found_entities = []
        
        for main_entity, variations in RAMAYANA_ENTITIES.items():
            for variation in variations:
                if variation in text_lower:
                    found_entities.append(main_entity)
                    break
        
        return list(set(found_entities))
    
    def _build_index(self):
        """Build inverted index for fast retrieval"""
        for chunk_idx, chunk in enumerate(self.chunks):
            # Index words
            words = re.findall(r'\b\w+\b', chunk["text"].lower())
            for word in words:
                if len(word) > 2 and word not in _english_stopwords_set:
                    self.word_to_chunks[word].add(chunk_idx)
            
            # Index entities
            for entity in chunk.get("entities", []):
                self.entity_to_chunks[entity].add(chunk_idx)
    
    def _get_keywords(self, claim: str) -> List[str]:
        """Extract keywords with entity expansion"""
        words = re.findall(r'\b\w+\b', claim.lower())
        keywords = [word for word in words if word not in _english_stopwords_set and len(word) > 2]
        
        # Expand with entity variations
        enhanced_keywords = list(keywords)
        for keyword in keywords:
            for main_entity, variations in RAMAYANA_ENTITIES.items():
                if keyword in variations:
                    enhanced_keywords.extend(variations[:3])  # Add top 3 variations
                    break
        
        return list(dict.fromkeys(enhanced_keywords))[:TOP_N_KEYWORDS_FOR_RETRIEVAL]
    
    def _is_core_fact(self, claim: str) -> bool:
        """Check if claim is a core Ramayana fact"""
        claim_lower = claim.lower()
        return any(fact in claim_lower for fact in CORE_RAMAYANA_FACTS)
    
    def retrieve_context(self, claim: str) -> Tuple[str, List[Dict]]:
        """Retrieve relevant context for claim"""
        if not self.chunks:
            return "", []
        
        entities = self._extract_entities(claim)
        keywords = self._get_keywords(claim)
        all_terms = set(entities + keywords)
        
        # Score chunks
        scored_chunks = []
        for idx, chunk in enumerate(self.chunks):
            chunk_text = chunk["text"].lower()
            
            # Keyword matching
            keyword_score = sum(2.0 for term in all_terms if term in chunk_text)
            
            # Entity matching with context
            entity_score = 0
            for entity in entities:
                if entity in chunk_text:
                    # Boost if entity appears with relationship words
                    relationship_context = any(rel in chunk_text for rel in ["with", "and", "to", "from"])
                    entity_score += 3.0 if relationship_context else 1.5
            
            # Source priority (supplementary gets boost)
            source_boost = 2.0 if "supplementary" in chunk["source"].lower() else 0.0
            
            total_score = keyword_score + entity_score + source_boost
            scored_chunks.append((total_score, idx, chunk))
        
        # Get top chunks
        scored_chunks.sort(reverse=True)
        top_chunks = [chunk for _, _, chunk in scored_chunks[:MAX_RETRIEVED_CHUNKS]]
        
        # Create context string
        context_parts = []
        for i, chunk in enumerate(top_chunks[:3]):
            text = chunk["text"][:400]  # Limit for model context
            context_parts.append(f"[Source {i+1}]: {text}")
        
        context_str = "\n\n".join(context_parts)
        return context_str, top_chunks

class EnhancedClassifier:
    """Enhanced classifier with RAG integration"""
    
    def __init__(self, model_path=None):
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.rag_system = RamayanaRAGSystem()
        self.classification_cache = {}
        
    def initialize(self):
        """Initialize the classifier"""
        install_requirements()
        self._load_model(self.model_path)
        self.rag_system.load_reference_data()
        
    def _load_model(self, model_path=None):
        """Load Mistral model from local path first, then HuggingFace as fallback"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try local paths first
        possible_local_paths = [
            model_path,
            "./downloaded_model",  # Your specific folder
            "./downloaded_model/teknium--OpenHermes-2.5-Mistral-7B",
            "./teknium--OpenHermes-2.5-Mistral-7B",
            "./OpenHermes-2.5-Mistral-7B",
            "./models/downloaded_model",
            "./models/teknium--OpenHermes-2.5-Mistral-7B",
            "./models/OpenHermes-2.5-Mistral-7B",
            "downloaded_model",
            "teknium--OpenHermes-2.5-Mistral-7B",
            "OpenHermes-2.5-Mistral-7B"
        ]
        
        model_loaded = False
        
        # Try loading from local paths first
        for local_path in possible_local_paths:
            if local_path and os.path.exists(local_path):
                try:
                    logger.info(f"Found local model at: {local_path}")
                    logger.info("Loading tokenizer from local path...")
                    
                    # Load tokenizer first
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_path,
                        local_files_only=True,
                        padding_side='left',
                        use_fast=True
                    )
                    logger.info("✅ Tokenizer loaded successfully")
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    logger.info("Loading model from local path...")
                    
                    # Load model with more explicit settings
                    self.model = AutoModelForCausalLM.from_pretrained(
                        local_path,
                        local_files_only=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else "cpu",
                        trust_remote_code=False,
                        low_cpu_mem_usage=True
                    )
                    
                    logger.info(f"✅ Model loaded successfully from local path: {local_path}")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load from {local_path}")
                    logger.error(f"Error details: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    continue
        
        # Fallback to HuggingFace if local loading failed
        if not model_loaded:
            model_name = "teknium/OpenHermes-2.5-Mistral-7B"
            logger.info("⚠️  Local model not found or failed to load")
            logger.info("Checking what's in the downloaded_model folder...")
            
            if os.path.exists("./downloaded_model"):
                files = os.listdir("./downloaded_model")
                logger.info(f"Files in downloaded_model: {files[:10]}...")  # Show first 10 files
            
            logger.info(f"📥 Downloading model from HuggingFace: {model_name}")
            logger.info("This may take several minutes...")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    padding_side='left',
                    use_fast=True
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    trust_remote_code=False,
                    low_cpu_mem_usage=True
                )
                
                logger.info("✅ Model downloaded and loaded successfully from HuggingFace!")
                
            except Exception as e:
                logger.error(f"❌ Failed to load model from both local and HuggingFace: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                raise
    
    def _create_prompt(self, statement: str, context: str) -> str:
        """Create optimized prompt with context"""
        prompt = f"""<|im_start|>system
You are an expert Ramayana scholar with deep knowledge of Valmiki's Ramayana. Classify statements as TRUE, FALSE, or IRRELEVANT based on the provided context.

Context from Valmiki's Ramayana:
{context}

RULES:
- TRUE: Statement is factually correct according to the context
- FALSE: Statement contradicts the context or known facts
- IRRELEVANT: Statement is not related to Ramayana

Respond with exactly one word: TRUE, FALSE, or IRRELEVANT<|im_end|>
<|im_start|>user
Statement: {statement}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _model_classify(self, statement: str, context: str) -> Tuple[str, float, str]:
        """Classify using model with context"""
        if not self.model:
            return "ERROR", 0.0, "Model not available"
        
        try:
            prompt = self._create_prompt(statement, context)
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_part = response[len(prompt):].strip()
            
            # Extract classification
            answer = self._extract_label(answer_part)
            confidence = 0.85 if answer in ["TRUE", "FALSE", "IRRELEVANT"] else 0.6
            
            return answer, confidence, answer_part
            
        except Exception as e:
            logger.error(f"Model classification error: {e}")
            return "ERROR", 0.0, str(e)
    
    def _extract_label(self, response: str) -> str:
        """Extract label from model response"""
        response_upper = response.upper()
        
        for label in ["TRUE", "FALSE", "IRRELEVANT"]:
            if label in response_upper:
                return label
        
        # Fallback patterns
        if any(word in response_upper for word in ["CORRECT", "YES", "ACCURATE"]):
            return "TRUE"
        elif any(word in response_upper for word in ["INCORRECT", "NO", "WRONG"]):
            return "FALSE"
        else:
            return "IRRELEVANT"
    
    def _rule_based_classify(self, statement: str) -> Tuple[str, float, str]:
        """Enhanced rule-based classification with better logic"""
        statement_lower = statement.lower()
        
        # Check if Ramayana-related
        ramayana_terms = 0
        found_entities = []
        
        for main_entity, variations in RAMAYANA_ENTITIES.items():
            if any(var in statement_lower for var in variations):
                ramayana_terms += 1
                found_entities.append(main_entity)
        
        # Not Ramayana-related
        if ramayana_terms == 0 and 'ramayana' not in statement_lower and 'valmiki' not in statement_lower:
            return "IRRELEVANT", 0.9, "No Ramayana entities found"
        
        # Check for obviously false statements
        false_patterns = [
            ('rama.*five.*brother', 'Rama had 3 brothers, not 5'),
            ('sita.*ravana.*sister', 'Sita was not Ravana\'s sister'),
            ('hanuman.*rama.*son', 'Hanuman was not Rama\'s son'),
            ('bharata.*exile', 'Bharata did not go into exile'),
            ('ravana.*good.*king', 'Ravana was not a good king'),
            ('lakshmana.*sita.*married', 'Lakshmana was not married to Sita')
        ]
        
        for pattern, reason in false_patterns:
            if re.search(pattern, statement_lower):
                return "FALSE", 0.8, reason
        
        # Check for obviously true statements
        true_patterns = [
            ('rama.*son.*dasharatha', 'Rama was indeed Dasharatha\'s son'),
            ('sita.*wife.*rama', 'Sita was Rama\'s wife'),
            ('hanuman.*devotee.*rama', 'Hanuman was Rama\'s greatest devotee'),
            ('ravana.*ten.*head', 'Ravana had ten heads'),
            ('exile.*fourteen.*year', 'Exile was for 14 years'),
            ('hanuman.*ocean.*lanka', 'Hanuman crossed the ocean to Lanka'),
            ('bharata.*sandals.*throne', 'Bharata placed Rama\'s sandals on throne'),
            ('vibhishana.*ravana.*brother', 'Vibhishana was Ravana\'s brother')
        ]
        
        for pattern, reason in true_patterns:
            if re.search(pattern, statement_lower):
                return "TRUE", 0.8, reason
        
        # Check core facts with more sophisticated matching
        if self.rag_system._is_core_fact(statement):
            return "TRUE", 0.7, "Matches core Ramayana fact pattern"
        
        # Check for family relationships
        family_patterns = [
            ('dasharatha.*father', 'TRUE', 0.75),
            ('kaushalya.*mother.*rama', 'TRUE', 0.75),
            ('janaka.*father.*sita', 'TRUE', 0.75),
            ('sumitra.*mother.*lakshmana', 'TRUE', 0.75),
            ('kaikeyi.*mother.*bharata', 'TRUE', 0.75)
        ]
        
        for pattern, label, conf in family_patterns:
            if re.search(pattern, statement_lower):
                return label, conf, f"Family relationship: {pattern}"
        
        # Default for Ramayana-related but uncertain content
        if found_entities:
            return "IRRELEVANT", 0.6, f"Ramayana-related (entities: {', '.join(found_entities[:3])}) but needs verification"
        else:
            return "IRRELEVANT", 0.5, "Possibly Ramayana-related but uncertain"
    
    def classify_statement(self, statement: str) -> Dict[str, Any]:
        """Main classification method with RAG"""
        start_time = time.time()
        
        # Check cache
        cache_key = hash(statement)
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # RAG retrieval
        context, retrieved_chunks = self.rag_system.retrieve_context(statement)
        
        # Model classification
        if context and self.model:
            label, confidence, explanation = self._model_classify(statement, context)
        else:
            label, confidence, explanation = self._rule_based_classify(statement)
        
        # Apply fallbacks for core facts
        if self.rag_system._is_core_fact(statement) and label == "FALSE":
            if retrieved_chunks:
                label = "TRUE"
                confidence = 0.8
                explanation += " [Core fact with RAG evidence]"
        
        result = {
            'statement': statement,
            'label': label,
            'confidence': confidence,
            'explanation': explanation[:200],
            'model_used': "Mistral + RAG" if context else "Rule-based",
            'processing_time': time.time() - start_time,
            'chunks_retrieved': len(retrieved_chunks)
        }
        
        # Cache result
        self.classification_cache[cache_key] = result
        return result

def process_csv(input_file: str, output_file: str, model_path: str = None):
    """Process CSV file with enhanced classifier"""
    try:
        # Initialize classifier
        classifier = EnhancedClassifier(model_path)
        classifier.initialize()
        
        # Load data
        df = pd.read_csv(input_file)
        logger.info(f"Processing {len(df)} statements...")
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying"):
            statement = row['statement']
            result = classifier.classify_statement(statement)
            results.append(result)
            
            if idx % 10 == 0:
                logger.info(f"Processed {idx+1}/{len(df)} statements")
        
        # Save results
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        label_counts = output_df['label'].value_counts()
        avg_confidence = output_df['confidence'].mean()
        
        print("\n" + "="*50)
        print("CLASSIFICATION SUMMARY")
        print("="*50)
        print(f"Total statements: {len(output_df)}")
        print(f"Average confidence: {avg_confidence:.2f}")
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(output_df)*100:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Enhanced Ramayana Classifier with RAG')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--model-path', type=str, help='Path to local model directory')
    
    args = parser.parse_args()
    
    print("Enhanced Ramayana Classifier with RAG Integration")
    print("="*60)
    
    process_csv(args.input, args.output, args.model_path)

if __name__ == "__main__":
    main()