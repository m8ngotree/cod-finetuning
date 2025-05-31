import json
import openai
import time
import re
import os
from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

class GSM8KBaselineEvaluator:
    def __init__(self, api_key: str = None):
        """
        Initialize the evaluator with OpenAI API
        
        Args:
            api_key: OpenAI API key (if None, will try to load from environment)
        """
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not found. Please either:\n"
                    "1. Pass api_key parameter, or\n"
                    "2. Set OPENAI_API_KEY in your .env file, or\n"
                    "3. Set OPENAI_API_KEY environment variable"
                )
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Chain of Draft system message
        self.cod_system_message = (
            "Think step by step, but only keep a minimum draft for each thinking step, "
            "with 5 words at most. Write everything in a single line with steps separated "
            "by periods. Return only the numerical answer at the end of the response "
            "after a separator ####."
        )

    def extract_numerical_answer(self, response: str) -> Optional[str]:
        """Extract numerical answer from model response as string"""
        # Look for #### followed by number
        match = re.search(r'####\s*([-\d\./\$,]+)', response)
        if match:
            # Clean up the answer (remove $ and commas if present)
            answer_str = match.group(1).replace('$', '').replace(',', '').strip()
            return answer_str
        
        # Fallback: look for last number in response
        numbers = re.findall(r'[-\d\./]+', response)
        if numbers:
            answer_str = numbers[-1].replace('$', '').replace(',', '').strip()
            return answer_str
        
        return None

    def extract_ground_truth(self, solution: str) -> Optional[str]:
        """Extract ground truth answer from GSM8K solution as string"""
        match = re.search(r'####\s*([-\d\./\$,]+)', solution)
        if match:
            answer_str = match.group(1).replace('$', '').replace(',', '').strip()
            return answer_str
        return None

    def evaluate_single_example(self, 
                               question: str, 
                               ground_truth: str,
                               model: str,
                               system_message: str,
                               max_retries: int = 3) -> Dict:
        """Evaluate a single example"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": question}
                    ],
                    temperature=0,  # Greedy sampling for evaluation
                    max_tokens=300
                )
                
                model_response = response.choices[0].message.content.strip()
                predicted_answer = self.extract_numerical_answer(model_response)
                
                # Direct string comparison for exact match
                is_correct = False
                if predicted_answer is not None and ground_truth is not None:
                    is_correct = predicted_answer == ground_truth
                
                return {
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_response": model_response,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "attempt": attempt + 1
                }
                
            except Exception as e:
                if attempt == max_retries - 1:
                    return {
                        "question": question,
                        "ground_truth": ground_truth,
                        "model_response": f"Error: {str(e)}",
                        "predicted_answer": None,
                        "is_correct": False,
                        "attempt": attempt + 1
                    }
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None

    def evaluate_model(self, 
                      model: str,
                      system_message: str,
                      eval_name: str,
                      max_examples: Optional[int] = None,
                      output_file: Optional[str] = None) -> Dict:
        """Evaluate a model on GSM8K test set"""
        
        print(f"\n=== Evaluating {eval_name} ===")
        print(f"Model: {model}")
        print(f"System message: {system_message[:100]}...")
        
        # Load GSM8K test set
        dataset = load_dataset("gsm8k", "main")["test"]
        
        if max_examples:
            # Sample random examples for faster evaluation
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            dataset = dataset.select(indices)
        
        print(f"Evaluating on {len(dataset)} examples...")
        
        results = []
        correct_count = 0
        
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {eval_name}")):
            ground_truth = self.extract_ground_truth(example["answer"])
            if ground_truth is None:
                continue
            
            result = self.evaluate_single_example(
                question=example["question"],
                ground_truth=ground_truth,
                model=model,
                system_message=system_message
            )
            
            if result:
                results.append(result)
                if result["is_correct"]:
                    correct_count += 1
            
            # Rate limiting
            time.sleep(0.1)
        
        # Calculate metrics
        total_evaluated = len(results)
        accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
        
        evaluation_summary = {
            "eval_name": eval_name,
            "model": model,
            "system_message": system_message,
            "total_examples": total_evaluated,
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
        
        print(f"Results for {eval_name}:")
        print(f"  Accuracy: {accuracy:.3f} ({correct_count}/{total_evaluated})")
        print(f"  Total examples: {total_evaluated}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(evaluation_summary, f, indent=2)
            print(f"  Results saved to: {output_file}")
        
        return evaluation_summary

    def compare_models(self, 
                      baseline_model: str = "gpt-4o-mini-2024-07-18",
                      finetuned_model: Optional[str] = None,
                      max_examples: Optional[int] = None,
                      output_dir: str = "evaluation_results") -> Dict:
        """Compare baseline and fine-tuned models with Chain of Draft prompting only"""
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        evaluations = {}
        
        # 1. Baseline model with Chain of Draft prompting
        print("\n" + "="*80)
        print("BASELINE MODEL EVALUATION (Chain of Draft)")
        print("="*80)
        
        baseline_cod = self.evaluate_model(
            model=baseline_model,
            system_message=self.cod_system_message,
            eval_name="Baseline (Chain of Draft)",
            max_examples=max_examples,
            output_file=f"{output_dir}/baseline_cod_{timestamp}.json"
        )
        evaluations["baseline_cod"] = baseline_cod
        
        # 2. Fine-tuned model with Chain of Draft (if provided)
        if finetuned_model:
            print("\n" + "="*80)
            print("FINE-TUNED MODEL EVALUATION (Chain of Draft)")
            print("="*80)
            
            finetuned_cod = self.evaluate_model(
                model=finetuned_model,
                system_message=self.cod_system_message,
                eval_name="Fine-tuned (Chain of Draft)",
                max_examples=max_examples,
                output_file=f"{output_dir}/finetuned_cod_{timestamp}.json"
            )
            evaluations["finetuned_cod"] = finetuned_cod
        
        # Generate comparison report
        self.generate_comparison_report(evaluations, f"{output_dir}/comparison_report_{timestamp}.json")
        
        return evaluations

    def generate_comparison_report(self, evaluations: Dict, output_file: str):
        """Generate a detailed comparison report for Chain of Draft evaluations"""
        
        print("\n" + "="*80)
        print("CHAIN OF DRAFT COMPARISON REPORT")
        print("="*80)
        
        # Create summary table
        summary_data = []
        for eval_name, eval_data in evaluations.items():
            summary_data.append({
                "Evaluation": eval_data["eval_name"],
                "Model": eval_data["model"],
                "Accuracy": f"{eval_data['accuracy']:.3f}",
                "Correct": eval_data["correct_answers"],
                "Total": eval_data["total_examples"]
            })
        
        # Print summary table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Calculate fine-tuning improvement (if both models evaluated)
        if "baseline_cod" in evaluations and "finetuned_cod" in evaluations:
            baseline_acc = evaluations["baseline_cod"]["accuracy"]
            finetuned_acc = evaluations["finetuned_cod"]["accuracy"]
            improvement = ((finetuned_acc - baseline_acc) / baseline_acc) * 100 if baseline_acc > 0 else 0
            print(f"\nFine-tuning Improvement: {improvement:+.1f}%")
            print(f"Absolute Improvement: {finetuned_acc - baseline_acc:+.3f}")
        
        # Save comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_type": "Chain of Draft Comparison",
            "summary": summary_data,
            "detailed_evaluations": evaluations
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_file}")

    def analyze_errors(self, evaluation_results: Dict, output_file: Optional[str] = None):
        """Analyze common error patterns"""
        
        print("\n=== ERROR ANALYSIS ===")
        
        errors = []
        for result in evaluation_results["results"]:
            if not result["is_correct"]:
                errors.append({
                    "question": result["question"][:100] + "...",
                    "ground_truth": result["ground_truth"],
                    "predicted": result["predicted_answer"],
                    "response": result["model_response"][:200] + "..."
                })
        
        print(f"Total errors: {len(errors)}")
        print(f"Error rate: {len(errors) / len(evaluation_results['results']):.3f}")
        
        # Show first few errors
        print("\nFirst 5 errors:")
        for i, error in enumerate(errors[:5]):
            print(f"\nError {i+1}:")
            print(f"  Question: {error['question']}")
            print(f"  Expected: {error['ground_truth']}")
            print(f"  Got: {error['predicted']}")
            print(f"  Response: {error['response']}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(errors, f, indent=2)
            print(f"\nFull error analysis saved to: {output_file}")

def main():
    """Main evaluation script - Chain of Draft comparison only"""
    
    # Initialize evaluator
    evaluator = GSM8KBaselineEvaluator()
    
    # Configuration
    MAX_EXAMPLES = 100  # Set to None for full test set (1,319 examples)
    BASELINE_MODEL = "gpt-4o-mini-2024-07-18"
    FINETUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal:cod-4o-mini-1:BdKLVfzC"  # Set to your fine-tuned model ID when available
    
    print("GSM8K Chain of Draft Performance Evaluation")
    print("="*50)
    print(f"Baseline model: {BASELINE_MODEL}")
    print(f"Fine-tuned model: {FINETUNED_MODEL or 'Not specified'}")
    print(f"Max examples: {MAX_EXAMPLES or 'All (1,319)'}")
    print(f"Evaluation focus: Chain of Draft prompting only")
    
    # Run comparison
    evaluations = evaluator.compare_models(
        baseline_model=BASELINE_MODEL,
        finetuned_model=FINETUNED_MODEL,
        max_examples=MAX_EXAMPLES
    )
    
    # Analyze errors for baseline Chain of Draft
    print("\n" + "="*80)
    print("ERROR ANALYSIS - Baseline Chain of Draft")
    print("="*80)
    evaluator.analyze_errors(
        evaluations["baseline_cod"],
        "evaluation_results/baseline_cod_errors.json"
    )
    
    # Analyze errors for fine-tuned model if available
    if "finetuned_cod" in evaluations:
        print("\n" + "="*80)
        print("ERROR ANALYSIS - Fine-tuned Chain of Draft")
        print("="*80)
        evaluator.analyze_errors(
            evaluations["finetuned_cod"],
            "evaluation_results/finetuned_cod_errors.json"
        )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print("Results focus on Chain of Draft prompting comparison.")
    print("Check the 'evaluation_results' directory for detailed results.")

if __name__ == "__main__":
    main()