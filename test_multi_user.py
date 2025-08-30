#!/usr/bin/env python3
"""
Test script for the multi-user conversation routing system.
"""

import sys
import os
import time
import asyncio

# Add the realtalk module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtalk'))

from router import ConversationRouter, Turn, FeatureExtractor

def test_multi_user_addressing():
    """Test multi-user addressing detection"""
    print("=== Testing Multi-User Addressing ===")
    
    router = ConversationRouter()
    
    # Simulate active users
    active_users = ["Alice", "Bob", "Charlie"]
    router.active_speakers.update(active_users)
    
    test_cases = [
        # (speaker, text, expected_action, description)
        ("Alice", "hey dukebot, can you help me?", "speak", "Direct bot addressing"),
        ("Bob", "Alice, can you check this for me?", "ignore", "Human-to-human addressing"),
        ("Charlie", "what about the deployment issue?", "ignore", "Follow-up without active thread"),
        ("Alice", "Bob can you send me that file?", "ignore", "Human addressing with specific name"),
        ("Bob", "dukebot explain how Docker works", "speak", "Bot addressing with imperative"),
        ("Charlie", "nice kill Bob!", "ignore", "Gaming chatter to another user"),
    ]
    
    for speaker, text, expected_action, description in test_cases:
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
        
        status = "✅" if action == expected_action else "❌"
        print(f"{status} {speaker}: '{text}' → {action} ({reason}, score={score:.2f})")
        print(f"    Expected: {expected_action} | {description}")
        
        # Show human addressing features if present
        features = decision.get("features", {})
        human_addr = features.get("human_address", 0)
        if human_addr > 0:
            print(f"    Human addressing score: {human_addr:.2f}")
        
        print()

def test_thread_conversations():
    """Test multi-user thread conversations"""
    print("=== Testing Thread Conversations ===")
    
    router = ConversationRouter()
    
    # Start a conversation
    print("1. Alice starts a technical discussion")
    turn1 = Turn(
        user_id="user_alice",
        display_name="Alice", 
        text="dukebot, how do I set up continuous deployment?",
        confidence=1.0,
        timestamp=time.time()
    )
    
    decision1 = router.route_turn(turn1)
    print(f"   Alice: {decision1['action']} ({decision1.get('reason')})")
    
    if decision1["action"] == "speak":
        thread_id = decision1["thread_id"]
        print(f"   Created thread: {thread_id}")
        
        # Simulate bot response
        router.on_bot_reply(thread_id, "You can use GitHub Actions or GitLab CI. Which do you prefer?", "choice")
        print("   Bot replied with choice question")
        
        # Test Bob joining the conversation
        print("\n2. Bob joins the technical discussion") 
        turn2 = Turn(
            user_id="user_bob",
            display_name="Bob",
            text="I'd recommend GitHub Actions for this",
            confidence=1.0,
            timestamp=time.time() + 2
        )
        
        decision2 = router.route_turn(turn2)
        print(f"   Bob: {decision2['action']} ({decision2.get('reason')}, score={decision2.get('score', 0):.2f})")
        
        # Test Charlie with off-topic message
        print("\n3. Charlie with gaming chatter")
        turn3 = Turn(
            user_id="user_charlie", 
            display_name="Charlie",
            text="anyone want to queue for ranked?",
            confidence=1.0,
            timestamp=time.time() + 3
        )
        
        decision3 = router.route_turn(turn3)
        print(f"   Charlie: {decision3['action']} ({decision3.get('reason')}, score={decision3.get('score', 0):.2f})")
        
        # Test Alice following up with technical question
        print("\n4. Alice follows up on the technical discussion")
        turn4 = Turn(
            user_id="user_alice",
            display_name="Alice",
            text="what about testing the deployments?",
            confidence=1.0,
            timestamp=time.time() + 4
        )
        
        decision4 = router.route_turn(turn4)
        print(f"   Alice: {decision4['action']} ({decision4.get('reason')}, score={decision4.get('score', 0):.2f})")
        
    # Show thread status
    threads = router.get_active_threads()
    print(f"\n   Active threads: {len(threads)}")
    for thread in threads:
        participants = ", ".join(thread['participants'])
        print(f"     - Thread {thread['id'][-8:]}: {participants} (age: {thread['age_seconds']:.1f}s)")

def test_feature_extraction_with_users():
    """Test feature extraction with actual user names"""
    print("=== Testing Feature Extraction with User Context ===")
    
    extractor = FeatureExtractor()
    active_users = ["Alice", "Bob", "Maya", "Charlie"]
    
    test_cases = [
        ("Alice", "Maya, can you check the logs?", "Human addressing Maya"),
        ("Bob", "hey dukebot, show me the error", "Bot addressing"),
        ("Maya", "Alice what about the database?", "Human addressing Alice"),
        ("Charlie", "dukebot explain this code", "Bot addressing"),
        ("Alice", "Bob can you restart the service?", "Human addressing Bob"),
    ]
    
    for speaker, text, description in test_cases:
        turn = Turn(
            user_id=f"user_{speaker.lower()}",
            display_name=speaker,
            text=text,
            confidence=1.0,
            timestamp=time.time()
        )
        
        features = extractor.extract_features(turn, active_users)
        
        print(f"\n{speaker}: '{text}' ({description})")
        
        # Show top features
        top_features = {k: v for k, v in features.items() if v > 0.1}
        if top_features:
            for feature, value in sorted(top_features.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feature}: {value:.2f}")
        else:
            print("  No significant features")

def test_threshold_effects():
    """Test how different thresholds affect routing"""
    print("\n=== Testing Threshold Effects ===")
    
    # Test ambiguous case
    test_text = "can someone help me with this?"
    speaker = "TestUser"
    
    turn = Turn(
        user_id="test_user",
        display_name=speaker,
        text=test_text,
        confidence=1.0,
        timestamp=time.time()
    )
    
    thresholds = [
        (0.3, 0.2, 0.1, "Very permissive"),
        (0.55, 0.45, 0.2, "Default"),
        (0.8, 0.7, 0.3, "Very strict"),
    ]
    
    print(f"Testing with: '{test_text}'")
    
    for addr_thresh, followup_thresh, margin_thresh, description in thresholds:
        router = ConversationRouter()
        router.update_thresholds(addr_thresh, followup_thresh, margin_thresh)
        
        decision = router.route_turn(turn)
        
        print(f"  {description} ({addr_thresh}/{followup_thresh}/{margin_thresh}): "
              f"{decision['action']} ({decision.get('reason')}, score={decision.get('score', 0):.2f})")

def main():
    """Run all tests"""
    print("Testing Multi-User Conversation Routing System")
    print("=" * 60)
    
    try:
        test_feature_extraction_with_users()
        test_multi_user_addressing()
        test_thread_conversations()
        test_threshold_effects()
        
        print("\n" + "=" * 60)
        print("All tests completed! Check output above for any issues.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())