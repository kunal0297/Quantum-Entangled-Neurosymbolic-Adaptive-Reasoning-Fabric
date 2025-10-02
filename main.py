#!/usr/bin/env python3
"""
NEURAGRAPH (DNGE) - Dynamic Neurosymbolic Graph Evolution
Simple CLI Interface for Hackathon Submission
"""

import sys
import os
import pandas as pd
import argparse

# Add current directory to path
sys.path.append(os.getcwd())

def main():
    """Simple main entry point for NEURAGRAPH (DNGE)"""
    
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
    
    args = parser.parse_args()
    
    print("NEURAGRAPH (DNGE) - Dynamic Neurosymbolic Graph Evolution")
    print("Agentic Reasoning System for Logic Problem Solving")
    print("-" * 55)
    
    if args.demo:
        run_demo()
    else:
        process_questions(args.input, args.output, args.verbose)

def run_demo():
    """Run simple demonstration"""
    
    print("\nDemonstration Mode")
    print("-" * 20)
    
    demo_questions = [
        ("What is 25 + 17?", "42"),
        ("Is 24 an even number?", "True"),
        ("Which is greater: 15 or 8?", "15"),
        ("What is 5 squared?", "25")
    ]
    
    try:
        from final_dnge_system import FinalDNGESystem
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
    """Process questions from input file"""
    
    print(f"\nProcessing Mode")
    print(f"-" * 20)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    
    try:
        # Load data
        if not os.path.exists(input_file):
            print(f"Error: File '{input_file}' not found")
            return
        
        df = pd.read_csv(input_file)
        print(f"Loaded: {len(df)} questions")
        
        # Initialize system
        from final_dnge_system import FinalDNGESystem
        dnge = FinalDNGESystem()
        
        results = []
        high_confidence = 0
        
        for i, row in df.iterrows():
            # Get question
            if 'problem_statement' in row:
                question = row['problem_statement']
                topic = row.get('topic', 'General')
            elif 'question' in row:
                question = row['question']  
                topic = row.get('topic', 'General')
            else:
                print("Error: No question column found")
                return
            
            if verbose and i < 5:  # Show first 5 in verbose mode
                print(f"\nQ{i+1}: {question[:80]}...")
            elif (i + 1) % 20 == 0:  # Progress every 20 questions
                print(f"Progress: {i+1}/{len(df)}")
            
            try:
                result = dnge.solve_question(question, topic)
                answer = result['result']
                confidence = result['confidence']
                method = result['method']
                
                if confidence > 0.8:
                    high_confidence += 1
                
                if verbose and i < 5:
                    print(f"Answer: {answer}")
                    print(f"Confidence: {confidence:.3f}")
                
                results.append({
                    'id': row.get('id', i),
                    'question': question,
                    'answer': dnge.normalize(answer),
                    'confidence': confidence,
                    'method': method,
                    'topic': topic
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
                    'topic': topic
                })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        # Summary
        avg_confidence = results_df['confidence'].mean()
        success_rate = (results_df['confidence'] > 0.5).mean()
        high_conf_rate = high_confidence / len(results_df)
        
        print(f"\nResults Summary")
        print(f"-" * 20)
        print(f"Questions processed: {len(results_df)}")
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