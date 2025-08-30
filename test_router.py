#!/usr/bin/env python3
"""
Test script for the conversation routing system.
This tests the router logic independently of Discord/audio processing.
"""

import sys
import os
import time

# Add the realtalk module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtalk'))

from router import ConversationRouter, Turn, FeatureExtractor

def test_feature_extraction():
    """Test the feature extraction functionality"""
    print("=== Testing Feature Extraction ===")
    
    extractor = FeatureExtractor()
    
    test_cases = [
        ("hey dukebot, can you help me?", "direct bot addressing"),
        ("Maya, can you check this for me?", "human addressing"),
        ("what about the other option?", "discourse marker"),
        ("yes, that sounds good", "confirmation"),
        ("gg wp nice game", "gaming chatter"),
        ("how do I fix this error?", "technical question"),
        ("where is it?", "short question"),
    ]
    
    for text, description in test_cases:
        turn = Turn(
            user_id="test_user",
            display_name="TestUser",
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        features = extractor.extract_features(turn)
        print(f"\n'{text}' ({description}):")
        
        # Show top features
        top_features = {k: v for k, v in features.items() if v > 0.1}
        if top_features:
            for feature, value in sorted(top_features.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feature}: {value:.2f}")
        else:
            print("  No significant features")

def test_routing_decisions():
    """Test the routing decision logic"""
    print("\n\n=== Testing Routing Decisions ===")
    
    router = ConversationRouter()
    
    # Test cases with expected decisions
    test_cases = [
        ("hey dukebot, can you explain how this works?", "speak", "direct addressing"),
        ("dukebot help me debug this", "speak", "direct addressing with imperative"), 
        ("Maya, can you send me that file?", "ignore", "human addressing"),
        ("nice kill bro gg", "ignore", "gaming chatter"),
        ("what about the performance issue?", "ignore", "follow-up without active thread"),
        ("yes that's correct", "ignore", "confirmation without expectation"),
    ]
    
    for text, expected_action, description in test_cases:
        turn = Turn(
            user_id="test_user",
            display_name="TestUser",
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        decision = router.route_turn(turn)
        action = decision["action"]
        reason = decision.get("reason", "unknown")
        score = decision.get("score", 0)
        
        status = "[PASS]" if action == expected_action else "[FAIL]"
        print(f"\n{status} '{text}' ({description})")
        print(f"    Expected: {expected_action}, Got: {action} ({reason}, score={score:.2f})")
        
        if action != expected_action:
            print(f"    Features: {decision.get('features', {})}")

def test_thread_management():
    """Test thread creation and follow-up logic"""
    print("\n\n=== Testing Thread Management ===")
    
    router = ConversationRouter()
    
    # Start a conversation
    print("1. Starting conversation with bot")
    turn1 = Turn(
        user_id="user1",
        display_name="User1",
        text="dukebot, how do I deploy this application?",
        confidence=1.0,
        timestamp=time.time()
    )
    
    decision1 = router.route_turn(turn1)
    print(f"   Decision: {decision1['action']} ({decision1.get('reason')})")
    
    if decision1["action"] == "speak":
        thread_id = decision1["thread_id"]
        print(f"   Created thread: {thread_id}")
        
        # Simulate bot response
        router.on_bot_reply(thread_id, "You can deploy using Docker or direct upload. Which would you prefer?", "choice")
        print("   Bot replied with choice question")
        
        # Test follow-up
        print("\n2. Testing follow-up response")
        turn2 = Turn(
            user_id="user1",
            display_name="User1", 
            text="docker please",
            confidence=1.0,
            timestamp=time.time() + 1
        )
        
        decision2 = router.route_turn(turn2)
        print(f"   Decision: {decision2['action']} ({decision2.get('reason')}, score={decision2.get('score', 0):.2f})")
        
        # Test off-topic interruption
        print("\n3. Testing off-topic interruption")
        turn3 = Turn(
            user_id="user2",
            display_name="User2",
            text="hey anyone want to queue for ranked?",
            confidence=1.0,
            timestamp=time.time() + 2
        )
        
        decision3 = router.route_turn(turn3)
        print(f"   Decision: {decision3['action']} ({decision3.get('reason')}, score={decision3.get('score', 0):.2f})")
        
        # Test another user following up on the original topic
        print("\n4. Testing different user following up")
        turn4 = Turn(
            user_id="user3",
            display_name="User3",
            text="what about using kubernetes instead?",
            confidence=1.0,
            timestamp=time.time() + 3
        )
        
        decision4 = router.route_turn(turn4)
        print(f"   Decision: {decision4['action']} ({decision4.get('reason')}, score={decision4.get('score', 0):.2f})")
        
    # Show active threads
    threads = router.get_active_threads()
    print(f"\n   Active threads: {len(threads)}")
    for thread in threads:
        print(f"     - {thread['id'][-8:]}: {thread['participants']} (age: {thread['age_seconds']:.1f}s)")

def test_threshold_tuning():
    """Test threshold adjustment"""
    print("\n\n=== Testing Threshold Tuning ===")
    
    router = ConversationRouter()
    
    # Ambiguous case that might change behavior with different thresholds
    test_text = "can you help me with this?"
    
    turn = Turn(
        user_id="test_user",
        display_name="TestUser",
        text=test_text,
        confidence=1.0,
        timestamp=time.time()
    )
    
    thresholds = [
        (0.3, 0.3, 0.1, "Very permissive"),
        (0.55, 0.45, 0.2, "Default"),
        (0.8, 0.7, 0.3, "Very strict"),
    ]
    
    print(f"Testing with text: '{test_text}'")
    
    for addr_thresh, followup_thresh, margin_thresh, description in thresholds:
        router.update_thresholds(addr_thresh, followup_thresh, margin_thresh)
        decision = router.route_turn(turn)
        
        print(f"  {description} ({addr_thresh}/{followup_thresh}/{margin_thresh}): "
              f"{decision['action']} ({decision.get('reason')}, score={decision.get('score', 0):.2f})")

def main():
    """Run all tests"""
    print("Testing Conversation Routing System")
    print("=" * 50)
    
    try:
        test_feature_extraction()
        test_routing_decisions()
        test_thread_management()
        test_threshold_tuning()
        
        print("\n\n=== Test Summary ===")
        print("All tests completed! Check output above for any issues.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())