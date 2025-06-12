"""
GPT-2 + RAG INTEGRATION
Combine your fine-tuned GPT-2 with your sophisticated RAG system
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time
import re
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

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

# Use your existing configuration
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 300
MAX_RETRIEVED_CHUNKS = 8
TOP_N_KEYWORDS_FOR_RETRIEVAL = 15
DEBUG_RETRIEVAL = True

# Use your existing entity mappings
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
    'kumbhakarna': ['kumbhakarna', 'sleeping giant', 'ravana brother'],
    'ashoka vatika': ['ashoka vatika', 'ashok vatika', 'garden', 'lanka garden'],
    'bridge': ['bridge', 'setu', 'rama setu', 'sethu', 'ocean bridge'],
    'sanjeevani': ['sanjeevani', 'sanjivani', 'healing herb', 'mountain herb']
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

class GPT2RamayanaFactChecker:
    """
    Enhanced Ramayana Fact Checker using your fine-tuned GPT-2 + existing RAG system
    Keeps ALL your sophisticated retrieval logic, just changes the inference model
    """
    
    def __init__(self, gpt2_model_path: str = "./ramayana-gpt2-working-final"):
        self.gpt2_model_path = gpt2_model_path
        self._english_stopwords = _english_stopwords_set
        
        # Load your fine-tuned GPT-2 model
        self._load_gpt2_model()
        
        # Use your existing RAG setup
        self.reference_data = self._load_reference_data()
        self._build_chunk_index()
        
    def _load_gpt2_model(self):
        """Load your fine-tuned GPT-2 model"""
        try:
            print(f"Loading fine-tuned GPT-2 from: {self.gpt2_model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.gpt2_model_path,
                local_files_only=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.gpt2_model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            self.device = next(self.model.parameters()).device
            logger.info(f"✅ GPT-2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load GPT-2 model: {e}")
            raise

    # KEEP ALL YOUR EXISTING RAG METHODS UNCHANGED
    def _adaptive_semantic_chunking(self, text: str, source_filename: str) -> List[Dict[str, str]]:
        """Enhanced chunking that adapts to content structure."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_entities = set()
        
        # Detect chapter/section boundaries
        chapter_markers = ["chapter", "canto", "sarga", "kanda", "book", "part"]
        
        for i, sentence in enumerate(sentences):
            sentence_entities = self._extract_entities_from_text(sentence)
            
            # Check for natural break points
            is_chapter_boundary = any(marker in sentence.lower() for marker in chapter_markers)
            
            if (len(current_chunk) + len(sentence) > CHUNK_SIZE and current_chunk) or is_chapter_boundary:
                if current_chunk.strip():
                    chunks.append({
                        "text": current_chunk.strip(),
                        "source": source_filename,
                        "entities": list(current_entities),
                        "semantic_density": len(current_entities) / len(current_chunk.split()) if current_chunk else 0
                    })
                
                # Smart overlap based on entity continuity
                overlap_sentences = []
                for j in range(max(0, i-3), i):
                    if any(entity in sentences[j].lower() for entity in sentence_entities):
                        overlap_sentences.append(sentences[j])
                
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_entities = set(sentence_entities)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_entities.update(sentence_entities)
        
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "source": source_filename,
                "entities": list(current_entities),
                "semantic_density": len(current_entities) / len(current_chunk.split()) if current_chunk else 0
            })
        
        return chunks

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract Ramayana entities from text."""
        text_lower = text.lower()
        found_entities = []
        
        for main_entity, variations in RAMAYANA_ENTITIES.items():
            for variation in variations:
                if variation in text_lower:
                    found_entities.append(main_entity)
                    break
        
        return list(set(found_entities))

    def _build_chunk_index(self):
        """Build inverted index for faster retrieval."""
        self.word_to_chunks = defaultdict(set)
        self.entity_to_chunks = defaultdict(set)
        
        if self.reference_data.get("source_type") == "chunked_text_files":
            chunks = self.reference_data.get("chunks", [])
            
            for chunk_idx, chunk_data in enumerate(chunks):
                words = re.findall(r'\b\w+\b', chunk_data["text"].lower())
                for word in words:
                    if len(word) > 2 and word not in self._english_stopwords:
                        self.word_to_chunks[word].add(chunk_idx)
                
                entities = chunk_data.get("entities", [])
                for entity in entities:
                    self.entity_to_chunks[entity].add(chunk_idx)
        
        logger.info(f"Built index with {len(self.word_to_chunks)} words and {len(self.entity_to_chunks)} entities")

    def _get_enhanced_keywords_from_claim(self, claim: str) -> List[str]:
        """Enhanced keyword extraction with entity expansion."""
        words = re.findall(r'\b\w+\b', claim.lower())
        
        if self._english_stopwords:
            keywords = [word for word in words if word not in self._english_stopwords and len(word) > 2]
        else:
            keywords = [word for word in words if len(word) > 3]
        
        enhanced_keywords = list(keywords)
        for keyword in keywords:
            for main_entity, variations in RAMAYANA_ENTITIES.items():
                if keyword in variations:
                    enhanced_keywords.extend(variations)
                    break
        
        unique_keywords = list(dict.fromkeys(enhanced_keywords))[:TOP_N_KEYWORDS_FOR_RETRIEVAL]
        return unique_keywords

    def _is_core_ramayana_fact(self, claim: str) -> bool:
        """Check if this is a well-known core Ramayana fact."""
        claim_lower = claim.lower()
        for fact_pattern in CORE_RAMAYANA_FACTS:
            if fact_pattern in claim_lower:
                return True
        return False

    def _enhanced_hybrid_retrieval_with_priority(self, claim: str, all_chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enhanced retrieval with PRIORITY BOOST for supplementary book."""
        
        # Extract key entities and relationships
        entities = self._extract_entities_from_text(claim)
        keywords = self._get_enhanced_keywords_from_claim(claim)
        all_search_terms = set(entities + keywords)
        
        # Score chunks with PRIORITY BOOST
        scored_chunks = []
        for chunk in all_chunks:
            chunk_text = chunk["text"].lower()
            chunk_source = chunk["source"].lower()
            
            # 1. Direct keyword matching
            keyword_score = sum(2.0 for term in all_search_terms if term.lower() in chunk_text)
            
            # 2. Entity matching with relationship context
            entity_score = 0
            for entity in entities:
                if entity.lower() in chunk_text:
                    relationship_terms = ["with", "and", "to", "from", "by", "in", "at"]
                    entity_context = any(term in chunk_text for term in relationship_terms)
                    entity_score += 3.0 if entity_context else 1.5
            
            # 3. Source relevance scoring
            source_score = 0
            if any(term in chunk_source for term in ["bala", "ayodhya", "aranya", "kishkindha", "sundara", "yuddha"]):
                source_score = 2.0
            
            # 4. Semantic similarity
            semantic_score = sum(1.5 for keyword in keywords if keyword in chunk_text)
            
            # 5. Context window scoring
            context_score = 0
            if len(chunk_text.split()) > 50:
                context_score = 1.0
            
            # Calculate base score
            base_score = (
                keyword_score * 1.5 +
                entity_score * 2.0 +
                semantic_score * 1.2 +
                source_score * 1.0 +
                context_score * 0.5
            )
            
            # Apply priority boost for supplementary content
            if "supplementary" in chunk_source:
                priority_boost = base_score * 2.0 + 20.0  # Double score + flat boost
                final_score = base_score + priority_boost
            else:
                final_score = base_score
            
            scored_chunks.append({
                "chunk": chunk,
                "score": final_score,
                "is_supplementary": "supplementary" in chunk_source
            })
        
        # Sort by final score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Log summary information
        if DEBUG_RETRIEVAL:
            supplementary_in_top = sum(1 for c in scored_chunks[:MAX_RETRIEVED_CHUNKS] if c.get("is_supplementary", False))
            logger.info(f"Retrieval: {supplementary_in_top}/{MAX_RETRIEVED_CHUNKS} supplementary chunks in top results")
            logger.info(f"Top scores: {[round(c['score'], 2) for c in scored_chunks[:5]]}")
        
        return [c["chunk"] for c in scored_chunks[:MAX_RETRIEVED_CHUNKS]]

    # NEW: GPT-2 INFERENCE METHOD
    def _query_gpt2_with_context(self, claim: str, context: str) -> str:
        """Query your fine-tuned GPT-2 with retrieved context"""
        
        # Create prompt with context (like your RAG system provides)
        prompt = f"""Context from Valmiki's Ramayana:
{context}

You are a specialized Ramayana fact-checker based on Valmiki's Ramayana. Your task is to determine if statements about the Ramayana are factually correct.

INSTRUCTIONS:
- Respond with only one word: TRUE, FALSE, or IRRELEVANT
- TRUE: Statement is factually correct according to Valmiki's Ramayana
- FALSE: Statement contradicts the facts in Valmiki's Ramayana  
- IRRELEVANT: Statement is not related to the Ramayana

Statement: {claim}
Label:"""
        
        try:
            # Tokenize with length limit
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512  # Fit within GPT-2's context
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with optimal settings for your model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.3,  # Balanced randomness
                    do_sample=True,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error querying GPT-2: {e}")
            return "ERROR: Model inference failed"

    def _parse_gpt2_response(self, response: str, original_claim: str) -> Dict:
        """Parse GPT-2 response with fallback logic"""
        
        result = {
            "claim": original_claim,
            "verdict": "IRRELEVANT",
            "confidence": 0.7,
            "reasoning": response,
            "model_used": "Fine-tuned GPT-2 + RAG"
        }
        
        try:
            # Extract prediction (first word)
            predicted = response.split()[0] if response else "UNKNOWN"
            predicted = predicted.upper().replace(".", "").replace(",", "")
            
            # Validate prediction
            if predicted in ['TRUE', 'FALSE', 'IRRELEVANT']:
                result['verdict'] = predicted
                result['confidence'] = 0.8
            else:
                # Fallback: look for keywords in response
                response_lower = response.lower()
                if 'true' in response_lower:
                    result['verdict'] = 'TRUE'
                elif 'false' in response_lower:
                    result['verdict'] = 'FALSE'
                else:
                    result['verdict'] = 'IRRELEVANT'
                result['confidence'] = 0.6
            
            # Apply fallback for core facts (same as your Ollama system)
            if self._is_core_ramayana_fact(original_claim):
                if result['verdict'] == 'FALSE':
                    # Check if retrieved chunks support the claim
                    retrieved_chunks = self.reference_data.get("retrieved_chunks_for_current_claim", [])
                    claim_keywords = self._get_enhanced_keywords_from_claim(original_claim)
                    
                    supporting_evidence = False
                    for chunk in retrieved_chunks[:3]:
                        chunk_text = chunk['text'].lower()
                        keyword_matches = sum(1 for kw in claim_keywords if kw in chunk_text)
                        if keyword_matches >= 2:
                            supporting_evidence = True
                            break
                    
                    if supporting_evidence:
                        result['verdict'] = 'TRUE'
                        result['confidence'] = 0.8
                        result['reasoning'] += "\n\n[RAG FALLBACK: Core fact with supporting evidence]"
                        logger.info(f"Applied RAG fallback for: {original_claim[:50]}...")
            
        except Exception as e:
            logger.warning(f"Error parsing GPT-2 response: {e}")
            result['reasoning'] = f"Parse error: {e}\n\nRaw response: {response}"
        
        return result

    def fact_check(self, claim: str) -> Dict:
        """
        Main fact-checking method combining your RAG system with fine-tuned GPT-2
        """
        logger.info(f"RAG + GPT-2 fact-checking: '{claim[:50]}...'")
        
        start_time = time.time()
        
        # Use your existing retrieval system
        if self.reference_data.get("source_type") == "chunked_text_files":
            all_chunks = self.reference_data.get("chunks", [])
            retrieved_chunks = self._enhanced_hybrid_retrieval_with_priority(claim, all_chunks)
            self.reference_data["retrieved_chunks_for_current_claim"] = retrieved_chunks
        
        retrieval_time = time.time() - start_time
        
        # Generate context from retrieved chunks (same as your system)
        context_str = ""
        if self.reference_data.get("retrieved_chunks_for_current_claim"):
            chunks = self.reference_data["retrieved_chunks_for_current_claim"][:3]
            context_parts = []
            for i, chunk in enumerate(chunks):
                source = chunk.get("source", "Unknown")
                text = chunk['text'][:400]  # Fit in GPT-2 context
                context_parts.append(f"[Source {i+1}: {source}]\n{text}")
            context_str = "\n\n".join(context_parts)
        
        # NEW: Use GPT-2 instead of Ollama
        inference_start = time.time()
        response = self._query_gpt2_with_context(claim, context_str)
        inference_time = time.time() - inference_start
        
        # Parse response with your fallback logic
        result = self._parse_gpt2_response(response, claim)
        
        # Add metadata (same as your system)
        result["raw_response"] = response.strip()
        result["retrieval_time"] = round(retrieval_time, 3)
        result["inference_time"] = round(inference_time, 3)
        result["chunks_retrieved"] = len(self.reference_data.get("retrieved_chunks_for_current_claim", []))
        result["primary_source"] = (self.reference_data.get("retrieved_chunks_for_current_claim", [{}])[0].get("source", "none") 
                                   if self.reference_data.get("retrieved_chunks_for_current_claim") else "none")
        
        return result

    # KEEP YOUR EXISTING DATA LOADING METHODS
    def _load_reference_data(self) -> Dict:
        """Load Ramayana reference data with PRIORITY-BASED ordering."""
        RAMAYANA_BOOK_FILES = [
            Path("data/valmiki_ramayan_supplementary_knowledge.txt"),  # HIGHEST PRIORITY
            Path("data/valmiki_ramayan_bala_kanda_book1.txt"),
            Path("data/valmiki_ramayan_ayodhya_kanda_book2.txt"),
            Path("data/valmiki_ramayan_aranya_kanda_book3.txt"),
            Path("data/valmiki_ramayan_kishkindha_kanda_book4.txt"),
            Path("data/valmiki_ramayan_sundara_kanda_book5.txt"),
            Path("data/valmiki_ramayan_yuddha_kanda_book6.txt"),
        ]
        
        book_chunks = self._load_text_from_files_enhanced(RAMAYANA_BOOK_FILES)
        if book_chunks:
            return {"source_type": "chunked_text_files", "chunks": book_chunks}
        
        logger.warning("No text content loaded from book files.")
        return {"source_type": "minimal_facts", "content": {}}

    def _load_text_from_files_enhanced(self, file_paths: List[Path]) -> List[Dict[str, str]]:
        """Enhanced file loading with semantic chunking."""
        all_chunks = []
        files_loaded_count = 0
        
        for file_path in file_paths:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content.strip()) < 100:
                        logger.warning(f"File {file_path} appears to be empty or too short")
                        continue
                    
                    file_chunks = self._adaptive_semantic_chunking(content, file_path.name)
                    all_chunks.extend(file_chunks)
                    files_loaded_count += 1
                    logger.info(f"Enhanced chunking: {file_path} -> {len(file_chunks)} chunks")
                    
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        logger.info(f"Enhanced loading: {files_loaded_count} files -> {len(all_chunks)} total chunks")
        return all_chunks

    def test_retrieval(self, claim: str) -> Dict:
        """Test retrieval system for debugging."""
        all_chunks = self.reference_data.get("chunks", [])
        retrieved_chunks = self._enhanced_hybrid_retrieval_with_priority(claim, all_chunks)
        
        return {
            "claim": claim,
            "total_chunks": len(all_chunks),
            "retrieved_chunks": len(retrieved_chunks),
            "is_core_fact": self._is_core_ramayana_fact(claim),
            "top_chunks": [
                {
                    "source": chunk.get("source", "unknown"),
                    "preview": chunk["text"][:200] + "...",
                    "entities": chunk.get("entities", [])
                }
                for chunk in retrieved_chunks[:3]
            ]
        }

# Test function
def test_rag_gpt2_integration():
    """Test the RAG + GPT-2 integration"""
    
    print("🚀 TESTING RAG + GPT-2 INTEGRATION")
    print("=" * 50)
    
    try:
        # Initialize the integrated system
        fact_checker = GPT2RamayanaFactChecker()
        
        # Test cases
        test_cases = [
            "Hanuman leaped across the ocean to reach Lanka.",
            "Sita was the daughter of Ravana.", 
            "Ravana had ten heads.",
            "Einstein developed the theory of relativity."
        ]
        
        print("🧪 Testing with RAG context...")
        
        for i, claim in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {claim} ---")
            
            result = fact_checker.fact_check(claim)
            
            print(f"✓ Verdict: {result['verdict']}")
            print(f"✓ Confidence: {result['confidence']}")
            print(f"✓ Chunks retrieved: {result['chunks_retrieved']}")
            print(f"✓ Primary source: {result['primary_source']}")
            print(f"✓ Response: {result['raw_response'][:50]}...")
        
        print(f"\n🎉 RAG + GPT-2 INTEGRATION SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

if __name__ == "__main__":
    test_rag_gpt2_integration()