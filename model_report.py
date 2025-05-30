import argparse
import json
import logging
from pathlib import Path

try:
    from src.ollama_fact_checker import OllamaRamayanaFactChecker # Ensure this path is correct
except ModuleNotFoundError as e:
    if e.name == 'docopt':
        print(f"ModuleNotFoundError: The required library 'docopt' is not installed.")
        print(f"Please install it by running the following command in your terminal:")
        print(f"pip install docopt")
        print(f"The script 'src/ollama_fact_checker.py' uses 'docopt' for command-line argument parsing.")
        exit(1)
    else:
        # Re-raise the exception if it's a different ModuleNotFoundError
        raise
except ImportError as e:
    # Catch other ImportErrors, e.g., if OllamaRamayanaFactChecker class is not found
    print(f"ImportError: Could not import 'OllamaRamayanaFactChecker'.")
    print(f"Please ensure 'src/ollama_fact_checker.py' defines this class correctly.")
    print(f"Details: {e}")
    exit(1)

# Setup logging for the report script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_report")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Ramayana Fact Checker and Log Results")
    parser.add_argument("--model_name", type=str, default="llama3:8b", help="Ollama model name to use for fact checking.")
    parser.add_argument("--eval_data_file", type=str, required=True, help="Path to the evaluation data JSONL file.")
    parser.add_argument("--output_log_file", type=str, default="evaluation_results.jsonl", help="Path to save the detailed evaluation log.")

    args = parser.parse_args()

    logger.info(f"Initializing OllamaRamayanaFactChecker with model: {args.model_name}")
    try:
        fact_checker = OllamaRamayanaFactChecker(model_name=args.model_name)
    except Exception as e:
        logger.error(f"Failed to initialize OllamaRamayanaFactChecker: {e}")
        return

    eval_data_path = Path(args.eval_data_file)
    output_log_path = Path(args.output_log_file)

    if not eval_data_path.exists():
        logger.error(f"Evaluation data file not found: {eval_data_path}")
        return

    statements_to_evaluate = []
    try:
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    statements_to_evaluate.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON in {eval_data_path} at line {line_number}: {e}")
    except Exception as e:
        logger.error(f"Failed to read evaluation data file {eval_data_path}: {e}")
        return

    if not statements_to_evaluate:
        logger.info("No statements found in the evaluation file.")
        return

    logger.info(f"Loaded {len(statements_to_evaluate)} statements from {eval_data_path}")
    logger.info(f"Starting evaluation. Results will be logged to: {output_log_path}")

    correct_predictions = 0
    
    try:
        with open(output_log_path, 'w', encoding='utf-8') as log_f:
            for i, item in enumerate(statements_to_evaluate, 1):
                claim = item.get("statement")
                expected_label = item.get("label")

                if not claim or not expected_label:
                    logger.warning(f"Skipping item {i} due to missing 'statement' or 'label': {item}")
                    continue

                logger.info(f"Processing statement {i}/{len(statements_to_evaluate)}: \"{claim[:70]}...\" (Expected: {expected_label})")
                
                try:
                    result = fact_checker.fact_check(claim)
                    model_verdict = result.get("verdict", "ERROR_NO_VERDICT")
                    model_confidence = result.get("confidence", 0.0)
                    model_reasoning = result.get("reasoning", "No reasoning provided.")
                except Exception as e:
                    logger.error(f"Error fact-checking claim '{claim}': {e}")
                    model_verdict = "ERROR_DURING_FACT_CHECK"
                    model_confidence = 0.0
                    model_reasoning = f"Error during fact-checking: {e}"

                logger.info(f"  Model Verdict: {model_verdict} (Confidence: {model_confidence:.2f})")

                log_entry = {
                    "claim": claim,
                    "expected_label": expected_label,
                    "model_verdict": model_verdict,
                    "model_confidence": model_confidence,
                    "model_reasoning": model_reasoning,
                    "model_used": fact_checker.model_name 
                }
                log_f.write(json.dumps(log_entry) + '\n')

                if model_verdict == expected_label:
                    correct_predictions += 1
                else:
                    logger.info(f"    -> Incorrect: Expected {expected_label}, Got {model_verdict}")
    except Exception as e:
        logger.error(f"Failed to write to output log file {output_log_path}: {e}")
        return

    accuracy = (correct_predictions / len(statements_to_evaluate)) * 100 if statements_to_evaluate else 0
    
    summary_log_file_path = Path("evaluation_summary.txt")
    try:
        with open(summary_log_file_path, 'w', encoding='utf-8') as summary_f:
            summary_f.write(f"Evaluation Summary\n")
            summary_f.write(f"--------------------\n")
            summary_f.write(f"Model Used: {fact_checker.model_name}\n")
            summary_f.write(f"Evaluation Data: {eval_data_path.name}\n")
            summary_f.write(f"Total statements: {len(statements_to_evaluate)}\n")
            summary_f.write(f"Correct predictions: {correct_predictions}\n")
            summary_f.write(f"Accuracy: {accuracy:.2f}%\n")
            summary_f.write(f"Detailed results logged to: {output_log_path.resolve()}\n")
    except Exception as e:
        logger.error(f"Failed to write summary log file {summary_log_file_path}: {e}")


    logger.info("\nEvaluation Complete.")
    logger.info(f"Model Used: {fact_checker.model_name}")
    logger.info(f"Evaluation Data: {eval_data_path.name}")
    logger.info(f"Total statements: {len(statements_to_evaluate)}")
    logger.info(f"Correct predictions: {correct_predictions}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Detailed results have been logged to: {output_log_path.resolve()}")
    logger.info(f"A summary has also been saved to: {summary_log_file_path.resolve()}")


if __name__ == "__main__":
    main()
