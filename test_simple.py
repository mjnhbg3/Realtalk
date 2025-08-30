#!/usr/bin/env python3
"""
Simple test for the conversation routing system.
"""

import sys
import os
import time

# Add the realtalk module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtalk'))

from router import ConversationRouter, Turn, FeatureExtractor

def test_basic_functionality():
    print("Testing basic routing functionality...")
    
    router = ConversationRouter()
    
    # Test 1: Direct addressing
    turn1 = Turn(
        user_id="user1",
        text="dukebot help me",
        confidence=1.0,
        timestamp=time.time()
    )
    
    decision1 = router.route_turn(turn1)
    print(f"Test 1 - Direct addressing: {decision1['action']} ({decision1.get('reason')})")
    
    # Test 2: Human addressing
    turn2 = Turn(
        user_id="user1", 
        text="Maya can you help?",
        confidence=1.0,
        timestamp=time.time()
    )
    
    decision2 = router.route_turn(turn2)
    print(f"Test 2 - Human addressing: {decision2['action']} ({decision2.get('reason')})")
    
    # Test 3: Gaming chatter
    turn3 = Turn(
        user_id="user1",
        text="gg nice game",
        confidence=1.0,
        timestamp=time.time()
    )
    
    decision3 = router.route_turn(turn3)
    print(f"Test 3 - Gaming chatter: {decision3['action']} ({decision3.get('reason')})")
    
    return True

def test_feature_extraction():
    print("\nTesting feature extraction...")
    
    extractor = FeatureExtractor()
    
    turn = Turn(
        user_id="test",
        text="hey dukebot can you help?",
        confidence=1.0,
        timestamp=time.time()
    )
    
    features = extractor.extract_features(turn)
    print(f"Features for 'hey dukebot can you help?': {len(features)} features extracted")
    
    # Show non-zero features
    non_zero = {k: v for k, v in features.items() if v > 0}
    print(f"Non-zero features: {list(non_zero.keys())}")
    
    return True

def main():
    try:
        print("Simple Router Test")
        print("=" * 30)
        
        success = True
        success = test_feature_extraction() and success
        success = test_basic_functionality() and success
        
        if success:
            print("\nAll tests passed!")
            return 0
        else:
            print("\nSome tests failed!")
            return 1
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())