#!/usr/bin/env python3
"""
Test script for the stricter conversation routing system.
"""

import sys
import os
import time

# Add the realtalk module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtalk'))

from router import ConversationRouter, Turn

def test_stricter_routing():
    """Test the routing decision logic with stricter thresholds"""
    print("=== Testing Stricter Routing Decisions ===")
    
    router = ConversationRouter()
    router.update_thresholds(0.7, 0.6, 0.25)  # Stricter thresholds
    
    # Test cases with expected decisions  
    test_cases = [
        ("hey dukebot, can you explain how this works?", "speak", "direct addressing"),
        ("dukebot help me debug this", "speak", "direct addressing with imperative"), 
        ("Maya, can you send me that file?", "ignore", "human addressing"),
        ("nice kill bro gg", "ignore", "gaming chatter"),
        ("what about the performance issue?", "ignore", "follow-up without active thread"),
        ("yes that's correct", "ignore", "confirmation without expectation"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for text, expected_action, description in test_cases:
        # Clear any existing threads before each test
        router.threads = []
        
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
        if action == expected_action:
            passed += 1
            
        print(f"{status} '{text}' ({description})")
        print(f"    Expected: {expected_action}, Got: {action} ({reason}, score={score:.2f})")
        
        if action != expected_action:
            print(f"    Features: {decision.get('features', {})}")
            
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    return passed == total

if __name__ == "__main__":
    success = test_stricter_routing()
    sys.exit(0 if success else 1)