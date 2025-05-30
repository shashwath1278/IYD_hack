"""
Ollama-based Ramayana Fact Checker using Llama 3.8B
Works with locally hosted Ollama models for completely offline operation.
Implements a basic RAG approach with text chunking and keyword retrieval.
"""

import json
# pandas is not used in this restored version, can be removed if not needed elsewhere
# import pandas as pd 
import requests
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import time
import re # For keyword extraction

# Attempt to import nltk and download stopwords if not present
# This logger setup might conflict if model_report.py also sets up a basicConfig.
# It's generally better to configure logging once at the application entry point.
# However, for a module, just getting the logger is fine.
logger = logging.getLogger(__name__) # Get logger for this module

try:
    import nltk
    from nltk.corpus import stopwords
    # Check if 'stopwords' corpus is available, if not, download it
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError: # More specific exception
        logger.info("nltk stopwords not found. Downloading...")
        nltk.download('stopwords')
        logger.info("nltk stopwords downloaded successfully.")
    # Initialize stopwords after potential download
    _english_stopwords_set = set(stopwords.words('english'))
except ImportError:
    logger.warning("nltk library not found. Keyword extraction for RAG will be basic. Run: pip install nltk")
    _english_stopwords_set = set() # Fallback to an empty set
except LookupError: # This might occur if nltk.download fails or is interrupted
    logger.error("nltk stopwords lookup failed even after attempting download. Keyword extraction will be basic.")
    _english_stopwords_set = set()
except Exception as e: # Catch any other exception during nltk setup
    logger.error(f"An unexpected error occurred during NLTK setup: {e}. Keyword extraction will be basic.")
    _english_stopwords_set = set()


# Define the paths to the Ramayana book text files
RAMAYANA_BOOK_FILES = [
    Path("data/valmiki_ramayan_bala_kanda_book1.txt"),
    Path("data/valmiki_ramayan_ayodhya_kanda_book2.txt"),
    Path("data/valmiki_ramayan_aranya_kanda_book3.txt"),
    Path("data/valmiki_ramayan_kishkindha_kanda_book4.txt"),
    Path("data/valmiki_ramayan_sundara_kanda_book5.txt"),
    Path("data/valmiki_ramayan_yuddha_kanda_book6.txt"),
]

# RAG Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_RETRIEVED_CHUNKS = 5
TOP_N_KEYWORDS_FOR_RETRIEVAL = 10
DEBUG_RETRIEVAL = True

class OllamaRamayanaFactChecker:
    def __init__(self, model_name: str = "llama3:8b", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.api_url = f"{ollama_host}/api/generate"
        self._english_stopwords = _english_stopwords_set # Use the globally loaded stopwords
        
        self._test_connection()
        self.reference_data = self._load_reference_data()
        
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found. Available: {model_names}")
            else:
                raise ConnectionError(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama server at {self.ollama_host}: {e}")
            raise
    
    def _chunk_text(self, text: str, source_filename: str) -> List[Dict[str, str]]:
        """Splits text into overlapping chunks."""
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": source_filename, "start_char": start})
            if end >= text_len:
                break
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
        logger.info(f"Chunked {source_filename} into {len(chunks)} chunks.")
        return chunks

    def _load_text_from_files(self, file_paths: List[Path]) -> List[Dict[str, str]]:
        """Loads content from text files and chunks it."""
        all_chunks = []
        files_loaded_count = 0
        for file_path in file_paths:
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_chunks = self._chunk_text(content, file_path.name)
                    all_chunks.extend(file_chunks)
                    files_loaded_count += 1
                    logger.info(f"Successfully loaded and chunked text from {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading or chunking {file_path}: {e}. Skipping this file.")
            else:
                logger.warning(f"Text file not found: {file_path}. Skipping.")
        
        logger.info(f"Loaded and chunked text from {files_loaded_count} book files into {len(all_chunks)} total chunks.")
        return all_chunks

    def _get_keywords_from_claim(self, claim: str) -> List[str]:
        """Extracts significant keywords from a claim."""
        words = re.findall(r'\b\w+\b', claim.lower())
        if self._english_stopwords: # Check if stopwords set is not empty
            keywords = [word for word in words if word not in self._english_stopwords and len(word) > 2]
        else: 
            keywords = [word for word in words if len(word) > 3] 
        
        if not keywords: return [] # handle case where no keywords are found after filtering

        # Use nltk.FreqDist only if nltk was successfully imported and keywords exist
        if 'nltk' in globals() and nltk and keywords: # Check if nltk module is available
            keyword_freq = nltk.FreqDist(keywords)
            sorted_keywords = [kw for kw, _ in keyword_freq.most_common(TOP_N_KEYWORDS_FOR_RETRIEVAL)]
        else: # Fallback if nltk.FreqDist is not available or no keywords
            sorted_keywords = keywords[:TOP_N_KEYWORDS_FOR_RETRIEVAL]
        return sorted_keywords

    def _generate_search_queries(self, claim: str) -> List[str]:
        """Generates alternative search queries based on the claim using the LLM."""
        prompt = f"""Given the following claim about the Ramayana, generate 2-3 concise search queries or questions that would help find relevant text passages to verify or refute this claim.
Focus on key entities, actions, and potential areas of ambiguity.
Return *only* the search queries/questions, each on a new line. Do not include any other explanatory text or numbering unless it's part of the query itself.

Claim: "{claim}"

Search Queries/Questions:
"""
        response_text = self._query_ollama(prompt, max_tokens=100)
        raw_lines = [q.strip() for q in response_text.split('\n') if q.strip()]
        
        processed_queries = []
        for line in raw_lines:
            cleaned_line = re.sub(r"^\s*\d+\.\s*", "", line)
            if len(cleaned_line) > 10 and not cleaned_line.lower().startswith(("here are", "search queries", "sure, here are")):
                processed_queries.append(cleaned_line)
        
        if processed_queries:
            logger.info(f"Generated search queries for '{claim}': {processed_queries}")
        else:
            logger.warning(f"Could not generate or parse clean search queries for '{claim}'. Using original claim keywords only. Raw response: {response_text}")
        return processed_queries

    def _retrieve_relevant_chunks_with_details(self, claim: str, all_chunks: List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Retrieves relevant chunks with their details (text, source, score).
        """
        if not all_chunks:
            return []

        original_keywords = self._get_keywords_from_claim(claim)
        generated_queries = self._generate_search_queries(claim)
        
        combined_keywords = set(original_keywords)
        for query in generated_queries:
            combined_keywords.update(self._get_keywords_from_claim(query))
        
        final_keywords = list(combined_keywords)
        
        if not final_keywords:
            logger.info("No keywords extracted from claim or generated queries for retrieval.")
            return []

        logger.info(f"Using combined keywords for retrieval: {final_keywords}")

        scored_chunks_details = []
        for i, chunk_data in enumerate(all_chunks):
            chunk_text_lower = chunk_data["text"].lower()
            unique_keywords_found = 0
            for keyword in final_keywords: 
                if keyword in chunk_text_lower:
                    unique_keywords_found += 1
            
            if unique_keywords_found > 0:
                scored_chunks_details.append({
                    "score": unique_keywords_found, 
                    "text": chunk_data["text"], 
                    "source": chunk_data["source"], 
                    "start_char": chunk_data.get("start_char", 0), # Ensure start_char exists
                    "id": i
                })
        
        scored_chunks_details.sort(key=lambda x: x["score"], reverse=True)
        top_chunks_details = scored_chunks_details[:MAX_RETRIEVED_CHUNKS]

        if top_chunks_details:
            logger.info(f"Retrieved {len(top_chunks_details)} relevant chunks with details. Top sources: {[chunk['source'] for chunk in top_chunks_details]}")
        else:
            logger.info("No relevant chunks found for the claim based on keywords.")
        return top_chunks_details

    def _load_reference_data(self) -> Dict:
        """Loads Ramayana reference data."""
        book_chunks = self._load_text_from_files(RAMAYANA_BOOK_FILES)
        if book_chunks:
            return {"source_type": "chunked_text_files", "chunks": book_chunks}
        
        logger.warning("No text content loaded from book files. Falling back to minimal facts.")
        return {"source_type": "minimal_facts", "content": self._get_minimal_ramayana_facts()}
    
    def _get_minimal_ramayana_facts(self) -> Dict:
        """Minimal Ramayana facts for testing."""
        return {
            "characters": {
                "Rama": {"description": "Prince of Ayodhya", "father": "Dasharatha", "wife": "Sita"},
                "Sita": {"description": "Princess of Mithila", "father": "Janaka"},
                "Ravana": {"description": "Demon king of Lanka"},
                "Hanuman": {"description": "Devoted follower of Rama"}
            },
            "events": {
                "exile": "Rama was exiled for 14 years",
                "sita_abduction": "Sita was abducted by Ravana"
            }
        }
    
    def _query_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """Query the Ollama model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3, "top_p": 0.9}
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return ""
    
    def create_fact_checking_prompt(self, claim: str) -> str:
        """
        Creates an enhanced fact-checking prompt with critical instructions,
        step-by-step analysis guidelines, and examples for careful reasoning.
        """
        context_str = "No specific context was retrieved for this claim from the Ramayana texts."
        # This part assumes that retrieved_chunks_for_current_claim is populated by fact_check method
        if self.reference_data.get("source_type") == "chunked_text_files":
            retrieved_chunks_data = self.reference_data.get("retrieved_chunks_for_current_claim", [])
            if retrieved_chunks_data:
                retrieved_chunk_texts = [chunk["text"] for chunk in retrieved_chunks_data]
                context_str = "\n\n---\n\n".join(retrieved_chunk_texts)
        
        # Using the new prompt structure from user analysis
        prompt = f"""You are fact-checking this claim about the Ramayana: "{claim}"

CONTEXT FROM RAMAYANA TEXTS:
---
{context_str}
---

CRITICAL VERIFICATION RULES:
1. **EXACT MATCHING**: Every detail in the claim must match the context exactly.
   - If claim says "Seetha" but context says "Rama" for the same role/action → FALSE for that component.
   - If claim says "wore on feet" but context says "placed below feet" → FALSE for that component.
   - If claim specifies timing (e.g., "first night") and context doesn't specify or contradicts timing → INSUFFICIENT_DATA or FALSE for that component.

2. **COMPOUND CLAIMS**: Break down the claim into its core components (e.g., WHO did WHAT, to WHOM/WHAT, WHEN, WHERE, HOW, WHY).
   - ALL essential components of the claim must be supported by the context for an overall TRUE verdict.
   - If any essential component is contradicted → FALSE.
   - If any essential component is not mentioned or verifiable from context → INSUFFICIENT_DATA (unless general knowledge strongly refutes a simple assertion, then FALSE).

3. **AVOID ASSUMPTIONS**: 
   - "Following someone" does not automatically mean "celebrating with joy." Verify the emotion or action.
   - If context says "widows of Dasaratha" and the claim involves Kausalya, verify if Kausalya is one of them based on context or general Ramayana knowledge if context is silent.
   - Do not infer actions or relationships not explicitly stated or strongly implied by the context.

4. **TIMING MATTERS**: 
   - If the claim specifies a timing (e.g., "first night," "sixth night," "very next day"), this timing must be explicitly verifiable from the context.
   - If context describes a general action but doesn't confirm the specific timing in the claim, that component is INSUFFICIENT_DATA or FALSE if contradicted.

STEP-BY-STEP ANALYSIS (Your thought process):
1. Break the claim into its essential components: [List each part, e.g., Subject, Action, Object, Timing, Location].
2. For each component, search the CONTEXT for supporting or contradicting evidence. Note if evidence is missing.
3. Evaluate if ALL essential components are supported by the CONTEXT.
4. Determine the final VERDICT based on the component analysis and CRITICAL VERIFICATION RULES.

EXAMPLE REASONING (Illustrative - do not use this example's context for the actual claim):
Claim: "Bharata wore Rama's wooden sandals on his feet while ruling as regent"
Context Snippet (Hypothetical): "Bharata, with deep reverence, took Rama's wooden sandals. He placed them on the royal throne and governed the kingdom in Rama's name, drawing inspiration from the sandals."
Component Analysis:
- Subject: Bharata (Matches context)
- Action: wore on his feet (Context says "placed them on the royal throne", not "wore on his feet". This is a contradiction.)
- Object: Rama's wooden sandals (Matches context)
- Circumstance: while ruling as regent (Context says "governed the kingdom in Rama's name". This aligns.)
Evidence: Context states Bharata "placed them [sandals] on the royal throne".
Explanation: The claim's action "wore on his feet" is directly contradicted by the context, which states the sandals were "placed on the royal throne". Although other components align, the core action is different.
Verdict: FALSE

NOW ANALYZE THE GIVEN CLAIM: "{claim}"

Provide your response in the following format:
VERDICT: [TRUE/FALSE/INSUFFICIENT_DATA]
CONFIDENCE: [0.0-1.0]
COMPONENT_ANALYSIS: [Your breakdown of the claim's components and whether each is supported, contradicted, or not found in the context.]
EVIDENCE: [Quote specific supporting/contradicting text from the CONTEXT. If using general knowledge because context is insufficient, state "Based on general Ramayana knowledge..."]
EXPLANATION: [Your overall reasoning for the verdict, linking back to the component analysis and critical rules.]
"""
        return prompt
    
    def fact_check(self, claim: str) -> Dict:
        """
        Fact-check a claim about the Ramayana.
        
        Args:
            claim: The claim to verify
            
        Returns:
            Dictionary with verdict, confidence, reasoning, etc.
        """
        if self.reference_data.get("source_type") == "chunked_text_files":
            all_chunks = self.reference_data.get("chunks", [])
        else:
            all_chunks = []
        
        logger.info(f"Fact-checking claim: '{claim}'")
        logger.info(f"Using {len(all_chunks)} chunks from reference data for verification.")
        
        # Retrieve relevant chunks based on the claim
        top_chunks_details = self._retrieve_relevant_chunks_with_details(claim, all_chunks)
        if top_chunks_details:
            logger.info(f"Retrieved {len(top_chunks_details)} relevant chunks for the claim.")
        else:
            logger.info("No relevant chunks found for the claim.")
        
        # Enhance reference data with retrieved chunks for this specific claim
        self.reference_data["retrieved_chunks_for_current_claim"] = top_chunks_details
        
        prompt = self.create_fact_checking_prompt(claim)
        response = self._query_ollama(prompt, max_tokens=1000)
        
        logger.info(f"Received response from Ollama for the claim.")
        
        # Parse the response using the updated parsing logic
        parsed_result = self._parse_fact_check_response(response, claim)
        
        # Include raw response for transparency, but this can be removed if not needed
        parsed_result["raw_response"] = response.strip()
        
        return parsed_result
    
    def _parse_fact_check_response(self, response: str, original_claim: str) -> Dict:
        """Parse the model's fact-checking response, adapting to new REASONING field."""
        result = {
            "claim": original_claim, "verdict": "INSUFFICIENT_DATA", "confidence": 0.0,
            "component_analysis": {}, "evidence": "", "explanation": "", "reasoning": ""
        }
        lines = response.split('\n')
        current_key = None
        accumulated_value = []
        
        try:
            for line_idx, line in enumerate(lines):
                line_stripped = line.strip()
                if line_stripped.startswith('VERDICT:'):
                    if current_key and accumulated_value: result[current_key.lower()] = "\n".join(accumulated_value).strip()
                    verdict = line_stripped.replace('VERDICT:', '').strip()
                    if verdict in ['TRUE', 'FALSE', 'PARTIALLY_TRUE', 'INSUFFICIENT_DATA']: result['verdict'] = verdict
                    current_key = 'verdict_details'; accumulated_value = []
                elif line_stripped.startswith('REASONING:'):
                    if current_key and accumulated_value: result[current_key.lower()] = "\n".join(accumulated_value).strip()
                    reasoning_text = line_stripped.replace('REASONING:', '').strip()
                    accumulated_value = [reasoning_text] if reasoning_text else []
                    for next_line in lines[line_idx+1:]:
                        if next_line.strip().startswith('CONFIDENCE:') or next_line.strip().startswith('VERDICT:'): break
                        accumulated_value.append(next_line)
                    result['reasoning'] = "\n".join(accumulated_value).strip()
                    current_key = 'reasoning_processed'; accumulated_value = []
                elif line_stripped.startswith('CONFIDENCE:'):
                    if current_key and accumulated_value and current_key != 'reasoning_processed': result[current_key.lower()] = "\n".join(accumulated_value).strip()
                    conf_str = line_stripped.replace('CONFIDENCE:', '').strip()
                    try: result['confidence'] = min(max(float(conf_str), 0.0), 1.0)
                    except ValueError: logger.warning(f"Could not parse confidence: {conf_str}")
                    current_key = None; accumulated_value = []
                elif line_stripped.startswith('EVIDENCE:'):
                    if current_key and accumulated_value and current_key != 'reasoning_processed': result[current_key.lower()] = "\n".join(accumulated_value).strip()
                    result['evidence'] = line_stripped.replace('EVIDENCE:', '').strip()
                    current_key = 'evidence'; accumulated_value = [result['evidence']] if result['evidence'] else []
                elif line_stripped.startswith('EXPLANATION:'):
                     if current_key and accumulated_value and current_key != 'reasoning_processed': result[current_key.lower()] = "\n".join(accumulated_value).strip()
                     result['explanation'] = line_stripped.replace('EXPLANATION:', '').strip()
                     current_key = 'explanation'; accumulated_value = [result['explanation']] if result['explanation'] else []
                elif current_key and current_key not in ['reasoning_processed']:
                    accumulated_value.append(line)
            if current_key and accumulated_value and current_key not in ['reasoning_processed']: result[current_key.lower()] = "\n".join(accumulated_value).strip()
            if result.get('reasoning') == response and result.get('explanation') != response and result.get('explanation'): result['reasoning'] = result['explanation']
            if result['explanation'] == response and result['reasoning'] != response : result['explanation'] = result['reasoning']
            elif result['explanation'] == response and (result['verdict'] != "INSUFFICIENT_DATA" or result['confidence'] != 0.0):
                 if 'REASONING:' not in response and 'EXPLANATION:' not in response:
                    result['explanation'] = "Raw model response: " + response
                    if not result.get('reasoning') or result.get('reasoning') == response: result['reasoning'] = "Raw model response: " + response
        except Exception as e:
            logger.warning(f"Error parsing response: {e}. Raw response: '{response[:200]}...'")
        return result

    # --- Start of new debugging methods ---

    def debug_specific_claims(self):
        """Debug the specific problematic claims"""
        if self.reference_data.get("source_type") != "chunked_text_files" or not self.reference_data.get("chunks"):
            logger.error("Cannot run debug_specific_claims without loaded chunked text files.")
            return

        problem_claims = [
            "The Pushpaka aerial car was originally built by Ravana for his travels.",
            "Upon returning from exile, Rama embraced Bharata on his lap.",
            "Kausalya personally helped in adorning Seetha after Rama's return from exile."
        ]
        
        original_debug_retrieval_status = DEBUG_RETRIEVAL
        globals()['DEBUG_RETRIEVAL'] = False # Temporarily disable standard debug logging for this specific debug

        for claim in problem_claims:
            print(f"\n{'='*50}")
            print(f"DEBUGGING CLAIM: {claim}")
            print(f"{'='*50}")
            
            # 1. Show retrieved chunks
            # Use _retrieve_relevant_chunks_with_details and then extract text for display
            retrieved_chunks_details = self._retrieve_relevant_chunks_with_details(claim, self.reference_data["chunks"])
            print(f"\n--- RETRIEVED CHUNKS ({len(retrieved_chunks_details)}) ---")
            if retrieved_chunks_details:
                for i, chunk_data in enumerate(retrieved_chunks_details):
                    print(f"\nCHUNK {i+1} (Score: {chunk_data.get('score', 'N/A')}, Source: {chunk_data['source']}):")
                    print(f"Text: {chunk_data['text'][:300]}...")
                    print("-" * 30)
            else:
                print("No chunks retrieved for this claim.")
            
            # Store these specific chunks for the prompt generation for this debug call
            self.reference_data["retrieved_chunks_for_current_claim"] = retrieved_chunks_details
            
            # 2. Show the exact prompt sent to LLM
            prompt = self.create_fact_checking_prompt(claim)
            print(f"\n--- PROMPT TO LLM (First 500 chars) ---")
            print(prompt[:500] + "...")
            
            # 3. Show LLM's raw response
            response = self._query_ollama(prompt)
            print(f"\n--- LLM RAW RESPONSE ---")
            print(response)
            
            print(f"\n{'='*50}")
        
        globals()['DEBUG_RETRIEVAL'] = original_debug_retrieval_status # Restore original debug status

    def search_pushpaka_origin(self):
        """Specifically search for Pushpaka's true origin"""
        if self.reference_data.get("source_type") != "chunked_text_files" or not self.reference_data.get("chunks"):
            logger.error("Cannot run search_pushpaka_origin without loaded chunked text files.")
            return

        keywords = ['pushpaka', 'kubera', 'originally', 'built', 'owned', 'creator', 'viswakarma', 'brahma']
        
        matching_chunks = []
        for chunk_data in self.reference_data["chunks"]:
            chunk_lower = chunk_data["text"].lower()
            if any(keyword in chunk_lower for keyword in keywords):
                matching_chunks.append(chunk_data)
        
        print(f"\n--- SEARCHING FOR PUSHPAKA ORIGIN ---")
        print(f"Keywords: {keywords}")
        print(f"Found {len(matching_chunks)} chunks mentioning Pushpaka-related origin terms:")
        for i, chunk_data in enumerate(matching_chunks[:5]): # Show top 5 matches
            print(f"\nCHUNK {i+1}:")
            print(f"Source: {chunk_data['source']}")
            print(f"Text: {chunk_data['text'][:400]}...")
            print("-" * 50)

    def search_kausalya_seetha_adornment(self):
        """Search for the specific Kausalya-Seetha adornment passage"""
        if self.reference_data.get("source_type") != "chunked_text_files" or not self.reference_data.get("chunks"):
            logger.error("Cannot run search_kausalya_seetha_adornment without loaded chunked text files.")
            return
            
        # Keywords to find Kausalya and Seetha in proximity, related to adornment or post-return context
        # Looking for Kausalya specifically, not just any widow.
        primary_keywords = ['kausalya']
        secondary_keywords = ['seetha', 'sita', 'adorn', 'deck', 'ornaments', 'dress', 'attire', 'beautif']
        
        matching_chunks = []
        for chunk_data in self.reference_data["chunks"]:
            chunk_lower = chunk_data["text"].lower()  
            if any(pk in chunk_lower for pk in primary_keywords) and \
               any(sk in chunk_lower for sk in secondary_keywords):
                # Further check: ensure Kausalya is mentioned, not just "mother" or "queen" generally
                if 'kausalya' in chunk_lower:
                     matching_chunks.append(chunk_data)
        
        print(f"\n--- SEARCHING FOR KAUSALYA & SEETHA ADORNMENT ---")
        print(f"Primary Keywords: {primary_keywords}, Secondary Keywords: {secondary_keywords}")
        print(f"Found {len(matching_chunks)} chunks potentially relevant to Kausalya and Seetha adornment:")
        for i, chunk_data in enumerate(matching_chunks[:5]): # Show top 5 matches
            print(f"\nCHUNK {i+1}:")
            print(f"Source: {chunk_data['source']}")
            # Print more text for this specific search to get better context
            print(f"Text: {chunk_data['text']}") 
            print("-" * 50)

    # --- End of new debugging methods ---

# Removed the __main__ block with docopt as this file is a module.
# The model_report.py script will handle execution and argument parsing.