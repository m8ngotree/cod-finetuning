import json
import openai
import time
import re
import os
from datasets import load_dataset
from typing import List, Dict, Optional
import random
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GSM8KToCoDConverter:
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini-2024-07-18"):
        """
        Initialize the converter with OpenAI API
        
        Args:
            api_key: OpenAI API key (if None, will try to load from environment)
            model: Model to use for conversion
        """
        # Try to get API key from parameter, then environment, then raise error
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
        self.model = model
        
        # System message that matches CoD paper exactly
        self.system_message = "Convert the given detailed Chain-of-Thought solution into the following Chain of Draft format: Think step by step, but only keep a minimum draft for each thinking step, with exactly 5 words or fewer per step. Write everything in a single line with steps separated by periods. Return only the numerical answer (no units, no extra text) at the end after a separator ####."

    def extract_answer(self, solution: str) -> str:
        """Extract the final numerical answer from GSM8K solution"""
        # Look for #### followed by number (improved pattern)
        match = re.search(r'####\s*([-\d\./\$,]+)', solution)
        if match:
            # Clean up the answer (remove $ and commas if present)
            answer = match.group(1).replace('$', '').replace(',', '')
            return answer
        
        # Fallback: look for <<calculation=result>> pattern at the end
        matches = re.findall(r'<<[^>]*=([-\d\./]+)>>', solution)
        if matches:
            return matches[-1]
        
        # Last resort: extract last number (improved pattern)
        numbers = re.findall(r'[-\d\./]+', solution)
        return numbers[-1] if numbers else "0"

    def clean_original_solution(self, solution: str) -> str:
        """Clean the original solution by removing calculation annotations"""
        # Remove <<calculation>> patterns
        cleaned = re.sub(r'<<[^>]*>>', '', solution)
        # Remove #### and answer
        cleaned = re.sub(r'####.*$', '', cleaned, flags=re.MULTILINE)
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def clean_cod_response(self, cod_response: str) -> str:
        """Clean CoD response by removing newlines and extra whitespace"""
        if "####" in cod_response:
            reasoning, answer = cod_response.split("####", 1)
            reasoning = re.sub(r'\s+', ' ', reasoning.strip())
            # Clean the answer to be numerical only
            answer = self.clean_final_answer(answer.strip())
            return f"{reasoning} #### {answer}"
        else:
            return re.sub(r'\s+', ' ', cod_response.strip())
        
    def clean_final_answer(self, answer: str) -> str:
        """Extract only the numerical part of the answer"""
        # Remove common non-numerical suffixes
        answer = re.sub(r'\s*(dollars?|pounds?|feet|inches?|years?|days?|hours?|minutes?|seconds?|people|items?|books?|pieces?|cents?)', '', answer, flags=re.IGNORECASE)
        # Remove currency symbols and extract number
        answer = re.sub(r'[^\d\.\-\/]', '', answer)
        return answer.strip()

    def convert_single_example(self, question: str, solution: str) -> Optional[Dict]:
        """Convert a single GSM8K example to CoD format"""
        try:
            # Extract the final answer
            final_answer = self.extract_answer(solution)
            
            # Clean the original solution
            clean_solution = self.clean_original_solution(solution)
            
            # Call OpenAI API with system message
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": clean_solution}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=200
            )
            
            cod_response = response.choices[0].message.content.strip()

            # Clean the response to remove newlines
            cod_response = self.clean_cod_response(cod_response)
            
            # Validate the response has the correct format
            if "####" not in cod_response:
                print(f"Warning: No #### found in response for question: {question[:50]}...")
                cod_response += f" #### {final_answer}"
            
            return {
                "question": question,
                "original_solution": solution,
                "cod_solution": cod_response,
                "final_answer": final_answer
            }
            
        except Exception as e:
            print(f"Error converting example: {e}")
            return None

    def convert_dataset(self, 
                       subset: str = "train", 
                       max_examples: Optional[int] = None,
                       output_file: str = "gsm8k_cod.jsonl",
                       batch_size: int = 50) -> List[Dict]:
        """
        Convert the entire GSM8K dataset to CoD format
        
        Args:
            subset: Which subset to convert ("train" or "test")
            max_examples: Maximum number of examples to convert
            output_file: Output file path
            batch_size: Number of examples to process before saving checkpoint
        """
        # Load GSM8K dataset
        dataset = load_dataset("gsm8k", "main")[subset]
        
        if max_examples:
            # Sample random examples for experimentation
            indices = random.sample(range(len(dataset)), min(max_examples, len(dataset)))
            dataset = dataset.select(indices)
        
        converted_examples = []
        failed_examples = []
        
        output_path = Path(output_file)
        checkpoint_file = output_path.with_suffix('.checkpoint.jsonl')
        
        # Load existing checkpoint if available
        start_idx = 0
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                for line in f:
                    converted_examples.append(json.loads(line))
            start_idx = len(converted_examples)
            print(f"Resuming from checkpoint: {start_idx} examples already processed")
        
        print(f"Converting {len(dataset)} examples from GSM8K {subset} set...")
        print(f"Estimated time: {len(dataset) * 0.1 / 60:.1f} minutes")
        
        for i, example in enumerate(dataset):
            if i < start_idx:
                continue
                
            # Progress indicator every 100 examples
            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{len(dataset)} ({(i+1)/len(dataset)*100:.1f}%)")
            
            converted = self.convert_single_example(
                example["question"], 
                example["answer"]
            )
            
            if converted:
                converted_examples.append(converted)
                
                # Save checkpoint every batch_size examples
                if len(converted_examples) % batch_size == 0:
                    with open(checkpoint_file, 'a') as f:
                        f.write(json.dumps(converted) + '\n')
                    print(f"Checkpoint saved: {len(converted_examples)} examples completed")
            else:
                failed_examples.append(i)
            
            # Rate limiting - reduced for full dataset
            time.sleep(0.02)  # Faster for full dataset processing
        
        # Save final results
        with open(output_file, 'w') as f:
            for example in converted_examples:
                f.write(json.dumps(example) + '\n')
        
        # Clean up checkpoint file
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        print(f"Conversion complete!")
        print(f"Successfully converted: {len(converted_examples)} examples")
        print(f"Failed conversions: {len(failed_examples)} examples")
        if len(converted_examples) + len(failed_examples) > 0:
            print(f"Success rate: {len(converted_examples)/(len(converted_examples)+len(failed_examples))*100:.1f}%")
        
        return converted_examples

    def create_finetuning_format(self, 
                                cod_examples: List[Dict], 
                                output_file: str = "gsm8k_cod_finetune.jsonl"):
        """
        Convert CoD examples to OpenAI fine-tuning format
        """
        finetune_data = []
        
        for example in cod_examples:
            # Create the fine-tuning format with system message
            finetune_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most. Write everything in a single line with steps separated by periods. Return only the numerical answer at the end of the response after a separator ####."
                    },
                    {
                        "role": "user", 
                        "content": f"{example['question']}"
                    },
                    {
                        "role": "assistant", 
                        "content": example['cod_solution']
                    }
                ]
            }
            finetune_data.append(finetune_example)
        
        # Shuffle the training data to avoid easy-to-hard ordering artifacts
        random.shuffle(finetune_data)
        
        # Save in OpenAI fine-tuning format
        with open(output_file, 'w') as f:
            for example in finetune_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"Fine-tuning data saved to {output_file}")
        print(f"Total examples: {len(finetune_data)}")
        
        return finetune_data

    def validate_cod_quality(self, cod_examples: List[Dict], sample_size: int = 20):
        """
        Validate the quality of converted CoD examples
        """
        print("=== CoD Quality Validation ===")
        
        sample_examples = random.sample(cod_examples, min(sample_size, len(cod_examples)))
        
        word_counts = []
        step_counts = []
        
        for i, example in enumerate(sample_examples):
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {example['question'][:80]}...")
            print(f"CoD: {example['cod_solution']}")
            
            # Analyze CoD structure (improved step splitting)
            cod_text = example['cod_solution'].split('####')[0].strip()
            # Split on periods, semicolons, or newlines
            cod_steps = re.split(r'[.;\n]', cod_text)
            cod_steps = [step.strip() for step in cod_steps if step.strip()]
            
            step_counts.append(len(cod_steps))
            
            # Count words per step
            step_word_counts = [len(step.split()) for step in cod_steps]
            word_counts.extend(step_word_counts)
            
            print(f"Steps: {len(cod_steps)}, Avg words per step: {sum(step_word_counts)/len(step_word_counts) if step_word_counts else 0:.1f}")
        
        print(f"\n=== Overall Statistics ===")
        print(f"Average steps per problem: {sum(step_counts)/len(step_counts):.1f}")
        print(f"Average words per step: {sum(word_counts)/len(word_counts):.1f}")
        print(f"Max words per step: {max(word_counts) if word_counts else 0}")
        print(f"Steps with >5 words: {sum(1 for w in word_counts if w > 5)} / {len(word_counts)}")

# Usage for full dataset
def main():
    # Initialize converter
    converter = GSM8KToCoDConverter(
        model="gpt-4o-mini-2024-07-18"
    )
    
    print("=== Converting FULL GSM8K Training Set ===")
    print("This will process ~7,473 examples and take approximately 12-25 minutes")
    print("The process supports checkpointing, so you can resume if interrupted")
    
    # Convert the full training set
    cod_examples = converter.convert_dataset(
        subset="train",
        max_examples=None,  # Process all examples
        output_file="gsm8k_cod_train_full.jsonl",
        batch_size=100  # Save checkpoints more frequently
    )
    
    # Validate quality on larger sample
    print("\n=== Quality Validation ===")
    converter.validate_cod_quality(cod_examples, sample_size=50)
    
    # Create fine-tuning format
    print("\n=== Creating Fine-tuning Format ===")
    converter.create_finetuning_format(
        cod_examples, 
        "gsm8k_cod_finetune_train_full.jsonl"
    )
    
    print("\n=== Full Dataset Conversion Complete! ===")
    print(f"Training data: gsm8k_cod_train_full.jsonl ({len(cod_examples)} examples)")
    print(f"Fine-tuning data: gsm8k_cod_finetune_train_full.jsonl")
    
    # Optional: Also convert test set
    print("\n=== Converting Test Set ===")
    test_examples = converter.convert_dataset(
        subset="test",
        max_examples=None,
        output_file="gsm8k_cod_test_full.jsonl",
        batch_size=50
    )
    
    print(f"Test data: gsm8k_cod_test_full.jsonl ({len(test_examples)} examples)")
    print("\nAll conversions complete!")

if __name__ == "__main__":
    main()