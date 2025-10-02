#!/usr/bin/env python3
"""
Simple test script for NEURAGRAPH system
"""

import sys
import os
sys.path.append(os.getcwd())

def test_basic_functionality():
    """Test basic NEURAGRAPH functionality"""
    
    print("NEURAGRAPH System Test")
    print("=" * 30)
    
    try:
        # Test simple questions
        test_questions = [
            ("What is 15 + 25?", "40"),
            ("Is 24 an even number?", "True"),
            ("Which is greater: 12 or 8?", "12")
        ]
        
        print("Loading NEURAGRAPH system...")
        from final_dnge_system import FinalDNGESystem
        dnge = FinalDNGESystem()
        print("✓ System loaded successfully")
        
        correct = 0
        total = len(test_questions)
        
        print(f"\nTesting {total} questions...")
        print("-" * 30)
        
        for i, (question, expected) in enumerate(test_questions, 1):
            print(f"\nQ{i}: {question}")
            
            try:
                result = dnge.solve_question(question, "Mathematics")
                answer = str(result['result'])
                confidence = result['confidence']
                method = result['method']
                
                print(f"Answer: {answer}")
                print(f"Expected: {expected}")
                print(f"Confidence: {confidence:.3f}")
                print(f"Method: {method}")
                
                if answer.lower() == expected.lower():
                    print("✓ CORRECT")
                    correct += 1
                else:
                    print("✗ INCORRECT")
                    
            except Exception as e:
                print(f"✗ ERROR: {e}")
        
        print(f"\nTest Results:")
        print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
        
        if correct >= total * 0.7:  # 70% threshold
            print("✓ System working well!")
            return True
        else:
            print("⚠ System needs attention")
            return False
            
    except Exception as e:
        print(f"✗ System test failed: {e}")
        return False

def test_csv_processing():
    """Test CSV processing functionality"""
    
    print(f"\nCSV Processing Test")
    print("=" * 30)
    
    try:
        import pandas as pd
        
        # Check if test file exists
        if not os.path.exists("data/test.csv"):
            print("✗ data/test.csv not found")
            return False
        
        # Load test data
        df = pd.read_csv("data/test.csv")
        print(f"✓ Loaded {len(df)} questions from data/test.csv")
        
        # Test processing first 5 questions
        from final_dnge_system import FinalDNGESystem
        dnge = FinalDNGESystem()
        
        results = []
        high_confidence = 0
        
        print("Processing first 5 questions...")
        
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            question = row['problem_statement']
            topic = row.get('topic', 'General')
            
            print(f"\nQ{i+1}: {question[:60]}...")
            
            try:
                result = dnge.solve_question(question, topic)
                answer = result['result']
                confidence = result['confidence']
                
                print(f"Answer: {answer}")
                print(f"Confidence: {confidence:.3f}")
                
                if confidence > 0.8:
                    high_confidence += 1
                
                results.append({
                    'id': i,
                    'answer': dnge.normalize(answer),
                    'confidence': confidence
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'id': i,
                    'answer': 'ERROR',
                    'confidence': 0.0
                })
        
        # Save test results
        results_df = pd.DataFrame(results)
        results_df.to_csv("test_output.csv", index=False)
        
        avg_confidence = results_df['confidence'].mean()
        success_rate = (results_df['confidence'] > 0.5).mean()
        
        print(f"\nCSV Test Results:")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"High confidence: {high_confidence}/5")
        print(f"✓ Results saved to test_output.csv")
        
        return True
        
    except Exception as e:
        print(f"✗ CSV processing failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting NEURAGRAPH System Tests...\n")
    
    # Run tests
    basic_test = test_basic_functionality()
    csv_test = test_csv_processing()
    
    print(f"\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Basic functionality: {'✓ PASS' if basic_test else '✗ FAIL'}")
    print(f"CSV processing: {'✓ PASS' if csv_test else '✗ FAIL'}")
    
    if basic_test and csv_test:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 NEURAGRAPH system is ready for hackathon!")
    else:
        print("\n⚠ Some tests failed - check system")
    
    print("\nSystem test complete.")
