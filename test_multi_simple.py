#!/usr/bin/env python3
"""
Simple test for multi-user conversation routing system.
"""

import sys
import os
import time

# Add the realtalk module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtalk'))

from router import ConversationRouter, Turn, FeatureExtractor

def test_multi_user_addressing():
    """Test multi-user addressing detection"""
    print("=== Testing Multi-User Addressing ===")
    
    # Each test case creates its own router to avoid contamination
    
    test_cases = [
        # (speaker, text, expected_action, description)
        ("Alice", "hey dukebot, can you help me?", "speak", "Direct bot addressing"),
        ("Bob", "Alice, can you check this for me?", "ignore", "Human-to-human addressing"),
        ("Charlie", "what about the deployment issue?", "ignore", "Follow-up without active thread"),
        ("Alice", "Bob can you send me that file?", "ignore", "Human addressing with specific name"),
        ("Bob", "dukebot explain how Docker works", "speak", "Bot addressing with imperative"),
        ("Charlie", "nice kill Bob!", "ignore", "Gaming chatter to another user"),
    ]
    
    success_count = 0
    
    for speaker, text, expected_action, description in test_cases:
        # Create fresh router for each test to avoid thread pollution
        router = ConversationRouter()
        active_users = ["Alice", "Bob", "Charlie"] 
        router.active_speakers.update(active_users)
        turn = Turn(
            user_id=f"user_{speaker.lower()}",
            display_name=speaker,
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        decision = router.route_turn(turn)
        action = decision["action"]
        reason = decision.get("reason", "unknown")
        score = decision.get("score", 0)
        
        if action == expected_action:
            status = "PASS"
            success_count += 1
        else:
            status = "FAIL"
        
        print(f"{status}: {speaker} -> '{text}' -> {action} ({reason}, score={score:.2f})")
        print(f"      Expected: {expected_action} | {description}")
        
        # Show human addressing features if present
        features = decision.get("features", {})
        human_addr = features.get("human_address", 0)
        bot_alias = features.get("bot_alias", 0)
        if human_addr > 0 or bot_alias > 0:
            print(f"      Features: human_address={human_addr:.2f}, bot_alias={bot_alias:.2f}")
        
        print()
    
    print(f"Results: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_feature_extraction():
    """Test feature extraction with user context"""
    print("=== Testing Enhanced Feature Extraction ===")
    
    extractor = FeatureExtractor()
    active_users = ["Alice", "Bob", "Maya"]
    
    test_cases = [
        ("Alice", "Maya, can you check the logs?", ["human_address"]),
        ("Bob", "hey dukebot, show me the error", ["bot_alias"]),
        ("Maya", "Alice what about the database?", ["human_address"]),
    ]
    
    all_correct = True
    
    for speaker, text, expected_features in test_cases:
        turn = Turn(
            user_id=f"user_{speaker.lower()}",
            display_name=speaker,
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        features = extractor.extract_features(turn, active_users)
        
        print(f"{speaker}: '{text}'")
        
        # Check expected features are present
        feature_check = True
        for expected_feature in expected_features:
            if features.get(expected_feature, 0) <= 0.1:
                feature_check = False
                print(f"  MISSING: {expected_feature}")
        
        if feature_check:
            print("  PASS: All expected features detected")
        else:
            print("  FAIL: Missing expected features")
            all_correct = False
        
        # Show top features
        top_features = [(k, v) for k, v in features.items() if v > 0.1]
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        for feature, value in top_features[:3]:
            print(f"    {feature}: {value:.2f}")
        print()
    
    return all_correct

def test_basic_routing():
    """Test basic routing logic"""
    print("=== Testing Basic Routing Logic ===")
    
    # Fresh router created for each test case
    
    test_cases = [
        ("Alice", "dukebot help me", "speak"),
        ("Bob", "Alice can you help?", "ignore"),
        ("Alice", "nice game", "ignore"),
    ]
    
    all_correct = True
    
    for speaker, text, expected in test_cases:
        # Fresh router for each test
        router = ConversationRouter()
        router.active_speakers.update(["Alice", "Bob"])
        turn = Turn(
            user_id=f"user_{speaker.lower()}",
            display_name=speaker, 
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        decision = router.route_turn(turn)
        action = decision["action"]
        
        if action == expected:
            print(f"PASS: '{text}' -> {action}")
        else:
            print(f"FAIL: '{text}' -> {action} (expected {expected})")
            all_correct = False
    
    return all_correct

def main():
    """Run all tests"""
    print("Testing Multi-User Conversation Routing System")
    print("=" * 50)
    
    try:
        test1_pass = test_feature_extraction()
        test2_pass = test_basic_routing() 
        test3_pass = test_multi_user_addressing()
        
        print("=" * 50)
        if test1_pass and test2_pass and test3_pass:
            print("SUCCESS: All tests passed!")
            return 0
        else:
            print("FAILURE: Some tests failed!")
            return 1
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())