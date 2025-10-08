#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import re
import time
import argparse
from typing import Dict, Any, List, Tuple, Optional

# Add current directory to path
sys.path.append(os.getcwd())

# Import FinalDNGESystem here to make it available globally
try:
    from final_dnge_system import FinalDNGESystem
except ImportError:
    print("Error: Could not import FinalDNGESystem. Ensure final_dnge_system.py exists.")
    sys.exit(1)


def main():
    
    parser = argparse.ArgumentParser(
        description="NEURAGRAPH (DNGE) - Agentic Reasoning System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--input', '-i', type=str, default='data/test.csv',
                        help='Input CSV file (default: data/test.csv)')
    parser.add_argument('--output', '-o', type=str, default='results.csv',
                        help='Output CSV file (default: results.csv)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demonstration')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    
    # --- NEW TRAINING ARGUMENTS ---
    parser.add_argument('--train', '-t', type=str,
                        help='Input CSV file for training (e.g., data/train.csv). Runs training mode.')
    parser.add_argument('--model-output', type=str, default='final_dnge_system_trained.pth',
                        help='Path to save the trained model/weights.')
    # -----------------------------
    
    args = parser.parse_args()
    
    print("NEURAGRAPH (DNGE) - Dynamic Neurosymbolic Graph Evolution")
    print("Agentic Reasoning System for Logic Problem Solving")
    print("-" * 55)
    
    if args.train:
        run_training(args.train, args.model_output, args.verbose)
    elif args.demo:
        run_demo()
    else:
        process_questions(args.input, args.output, args.verbose)


def run_training(train_file: str, model_output_path: str, verbose: bool):
    
    print("\nTraining Mode")
    print("-" * 20)
    print(f"Input Data: {train_file}")
    print(f"Model Output: {model_output_path}")

    if not os.path.exists(train_file):
        print(f"Error: Training file '{train_file}' not found.")
        return

    try:
        df_train = pd.read_csv(train_file)
        print(f"Loaded {len(df_train)} training samples.")

        dnge = FinalDNGESystem()
        
        start_time = time.time()
        
        print("Starting system training...")
        dnge.train_system(df_train, verbose=verbose)
        
        end_time = time.time()
        
        dnge.save_model(model_output_path)
        
        print(f"Training successfully completed in {end_time - start_time:.2f} seconds.")
        print(f"Trained model parameters saved to {model_output_path}")

    except Exception as e:
        print(f"Training failed: {type(e).__name__}: {e}")
        

def run_demo():
    
    print("\nDemonstration Mode")
    print("-" * 20)
    
    demo_questions = [
        ("What is 25 + 17?", "42"),
        ("Is 24 an even number?", "True"),
        ("Which is greater: 15 or 8?", "15"),
        ("What is 5 squared?", "25")
    ]
    
    try:
        dnge = FinalDNGESystem()
        
        correct = 0
        total = len(demo_questions)
        
        for i, (question, expected) in enumerate(demo_questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                result = dnge.solve_question(question, "Mathematics")
                answer = result['result']
                confidence = result['confidence']
                
                print(f"Answer: {answer}")
                print(f"Confidence: {confidence:.2f}")
                print(f"Expected: {expected}")
                
                if str(answer).lower() == expected.lower():
                    print("✓ Correct")
                    correct += 1
                else:
                    print("✗ Incorrect")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        print(f"\nDemo Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
    except Exception as e:
        print(f"Demo failed: {e}")


def process_questions(input_file: str, output_file: str, verbose: bool = False):
    
    print(f"\nProcessing Mode")
    print(f"-" * 20)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    try:
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            return
        
        df = pd.read_csv(input_file)
        print(f"Loaded: {len(df)} questions")
        
        dnge = FinalDNGESystem()
        
        results = []
        total_confidence = 0.0
        success_count = 0
        high_confidence_count = 0
        
        for i, row in df.iterrows():
            if 'problem_statement' in row:
                question = row['problem_statement']
                topic = row.get('topic', 'General')
                expected_answer = row.get('target_answer', 'N/A') 
            elif 'question' in row:
                question = row['question']  
                topic = row.get('topic', 'General')
                expected_answer = row.get('target_answer', 'N/A')
            else:
                print("Error: No question column found")
                return
            
            if verbose and i < 5:
                print(f"\nQ{i+1}: {question[:80]}...")
            elif (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{len(df)}")
            
            try:
                result = dnge.solve_question(question, topic)
                answer = result['result']
                confidence = result['confidence']
                method = result['method']
                
                total_confidence += confidence
                
                if confidence > 0.5:
                    success_count += 1
                
                if confidence > 0.8:
                    high_confidence_count += 1
                
                if verbose and i < 5:
                    print(f"Answer: {answer}")
                    print(f"Confidence: {confidence:.3f}")
                
                results.append({
                    'id': row.get('id', i),
                    'question': question,
                    'answer': dnge.normalize(answer),
                    'confidence': confidence,
                    'method': method,
                    'topic': topic,
                    'expected_answer': expected_answer
                })
                
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                
                results.append({
                    'id': row.get('id', i),
                    'question': question,
                    'answer': 'ERROR',
                    'confidence': 0.0,
                    'method': 'error',
                    'topic': topic,
                    'expected_answer': expected_answer
                })
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        total_questions = len(results_df)
        avg_confidence = total_confidence / total_questions
        success_rate = success_count / total_questions
        high_conf_rate = high_confidence_count / total_questions
        
        print(f"\nResults Summary")
        print(f"-" * 20)
        print(f"Questions processed: {total_questions}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Success rate (>0.5): {success_rate:.1%}")
        print(f"High confidence (>0.8): {high_conf_rate:.1%}")
        print(f"Results saved: {output_file}")
        
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
