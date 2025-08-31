"""
Sophisticated conversation routing system for RealTalk

Implements per-speaker STT with intelligent routing based on:
- Addressing signals (bot mentions, imperatives, etc.)
- Follow-up affinity (topic similarity, discourse markers)
- Thread management with multiple concurrent threads
- Context-aware decision making

This replaces the simple wake-word detection with a more nuanced
understanding of when the bot should respond.
"""

import asyncio
import logging
import time
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
try:
    import numpy as np
except ImportError:
    # Fallback for testing without numpy
    class MockLinalg:
        def norm(self, x): return (sum(i**2 for i in x))**0.5
    
    class MockRandom:
        def seed(self, x): pass
        def randn(self, n): return [0.1]*n
    
    class MockNumpy:
        def __init__(self):
            self.linalg = MockLinalg()
            self.random = MockRandom()
        
        def dot(self, a, b): return sum(x*y for x,y in zip(a,b))
        def exp(self, x): return 2.718**x
        def log(self, x): return __import__('math').log(x)
        def randn(self, n): return [0.1]*n
    np = MockNumpy()

log = logging.getLogger("red.realtalk.router")

@dataclass
class Turn:
    """Represents a single utterance from a user"""
    user_id: str
    display_name: str
    text: str
    confidence: float
    timestamp: float
    embedding: Optional[List[float]] = None

@dataclass
class Thread:
    """Represents an active conversation thread"""
    id: str
    topic_vec: List[float]
    last_bot_vec: Optional[List[float]] = None
    participants: Set[str] = field(default_factory=set)
    last_activity: float = field(default_factory=time.time)
    expects_reply: Optional[Dict[str, Any]] = None
    
    def add_participant(self, user_id: str):
        """Add a participant to this thread"""
        self.participants.add(user_id)
        
    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = time.time()

class FeatureExtractor:
    """Extracts features from text for routing decisions"""
    
    def __init__(self):
        # Bot addressing patterns
        self.bot_aliases = ["dukebot", "duke bot", "duke", "bot", "ai", "assistant"]
        self.imperative_verbs = [
            "explain", "tell", "show", "help", "define", "describe", "summarize",
            "find", "search", "look", "check", "analyze", "calculate", "convert",
            "create", "make", "build", "generate", "write", "code", "debug"
        ]
        
        # Discourse markers for follow-up detection
        self.discourse_markers = [
            "what about", "and", "yeah but", "but what", "so then", "also",
            "additionally", "furthermore", "however", "though", "actually",
            "wait", "hold on", "speaking of", "regarding", "about that"
        ]
        
        # Short confirmation patterns
        self.confirmations = [
            "yes", "yeah", "yep", "ok", "okay", "sure", "please", "go ahead",
            "continue", "right", "correct", "exactly", "true", "agreed"
        ]
        
        # Gameplay/casual chat keywords
        self.game_keywords = [
            "gg", "wp", "nice", "lol", "omg", "wtf", "boss", "raid", "pvp",
            "queue", "match", "game", "play", "stream", "clip", "kill", "death",
            "noob", "rekt", "pwned", "ez", "clutch", "frag"
        ]
        
        # Human addressing patterns
        self.name_pattern = re.compile(r'^(?:@?(\w+),?\s+|hey\s+(\w+),?\s+)', re.IGNORECASE)
        
    def extract_features(self, turn: Turn, active_users: Optional[List[str]] = None) -> Dict[str, float]:
        """Extract routing features from a turn"""
        text = turn.text.lower().strip()
        words = text.split()
        
        features = {}
        
        # A. Addressing signal features
        features.update(self._extract_addressing_features(text, words))
        
        # A2. Human addressing features (now with actual user names)
        features.update(self._extract_human_addressing_features(text, words, turn.display_name, active_users or []))
        
        # B. Follow-up features (will be computed with thread context)
        features.update(self._extract_discourse_features(text, words))
        
        # C. Short interjection features
        features.update(self._extract_interjection_features(text, words))
        
        # D. Topic classification features
        features.update(self._extract_topic_features(text, words))
        
        return features
    
    def _extract_addressing_features(self, text: str, words: List[str]) -> Dict[str, float]:
        """Extract features related to addressing the bot or humans"""
        features = {}
        
        # Bot alias mentions
        bot_alias_score = 0.0
        for alias in self.bot_aliases:
            if alias in text:
                bot_alias_score = max(bot_alias_score, 1.0)
                # Boost if at start of message
                if text.startswith(alias) or text.startswith(f"hey {alias}"):
                    bot_alias_score = 1.2
        features['bot_alias'] = bot_alias_score
        
        # @ mention detection (simplified)
        features['at_mention'] = 1.0 if '@' in text else 0.0
        
        # Second-person targeting with "you"
        you_score = 0.0
        if 'you' in words:
            you_idx = words.index('you')
            # Check for imperative context around "you"
            context = ' '.join(words[max(0, you_idx-2):you_idx+3])
            for verb in self.imperative_verbs:
                if verb in context:
                    you_score = 1.0
                    break
        features['you_targeting'] = you_score
        
        # Agent verbs (commands/requests)
        agent_verb_score = 0.0
        for verb in self.imperative_verbs:
            if verb in text:
                agent_verb_score = max(agent_verb_score, 0.8)
                # Boost if at start
                if text.startswith(verb) or any(text.startswith(f"{w} {verb}") for w in ["can", "could", "will", "would", "please"]):
                    agent_verb_score = 1.0
        features['agent_verbs'] = agent_verb_score
        
        # Human addressing detection (basic pattern)
        human_address_score = 0.0
        match = self.name_pattern.match(text)
        if match:
            human_address_score = 0.5  # Reduced since we have better detection below
        features['human_address_pattern'] = human_address_score
        
        return features
    
    def _extract_human_addressing_features(self, text: str, words: List[str], speaker: str, active_users: List[str]) -> Dict[str, float]:
        """Detect if speaker is addressing another human by name"""
        features = {}
        
        human_address_score = 0.0
        
        # Check for other user names in the text (case-insensitive)
        for user in active_users:
            if user.lower() != speaker.lower():  # Don't match self
                user_lower = user.lower()
                
                # Check various addressing patterns
                patterns = [
                    f"{user_lower},",           # "Maya,"
                    f"hey {user_lower}",        # "hey Maya" 
                    f"@{user_lower}",           # "@Maya"
                    f"{user_lower} can you",    # "Maya can you"
                    f"{user_lower} could you",  # "Maya could you"
                    f"{user_lower} will you",   # "Maya will you"
                    f"{user_lower} would you",  # "Maya would you"
                ]
                
                for pattern in patterns:
                    if pattern in text:
                        human_address_score = 1.0
                        break
                
                # Also check if user name appears at start of message
                if text.startswith(user_lower):
                    human_address_score = max(human_address_score, 0.8)
                
                if human_address_score > 0:
                    break
        
        features['human_address'] = human_address_score
        return features
    
    def _extract_discourse_features(self, text: str, words: List[str]) -> Dict[str, float]:
        """Extract features related to discourse continuation"""
        features = {}
        
        # Discourse markers
        discourse_score = 0.0
        for marker in self.discourse_markers:
            if marker in text:
                discourse_score = max(discourse_score, 0.8)
                # Boost if at start of message
                if text.startswith(marker):
                    discourse_score = 1.0
        features['discourse_markers'] = discourse_score
        
        # Question continuation patterns
        question_score = 0.0
        if text.endswith('?'):
            question_score = 0.6
            # Check for follow-up question patterns
            follow_up_patterns = ["what about", "how about", "what if", "why", "where", "when", "how"]
            for pattern in follow_up_patterns:
                if pattern in text:
                    question_score = 0.9
                    break
        features['question_continuation'] = question_score
        
        # Ellipsis or incomplete thought patterns
        ellipsis_score = 0.0
        if '...' in text or text.endswith(',') or len(words) <= 3:
            ellipsis_score = 0.5
        features['ellipsis'] = ellipsis_score
        
        return features
    
    def _extract_interjection_features(self, text: str, words: List[str]) -> Dict[str, float]:
        """Extract features for short interjections and confirmations"""
        features = {}
        
        # Short confirmation detection
        confirmation_score = 0.0
        if len(words) <= 3:
            for conf in self.confirmations:
                if conf in words:
                    confirmation_score = 1.0
                    break
        features['confirmation'] = confirmation_score
        
        # Very short questions
        short_question_score = 0.0
        if len(words) <= 3 and text.endswith('?'):
            short_question_score = 1.0
        features['short_question'] = short_question_score
        
        # Single word responses
        single_word_score = 1.0 if len(words) == 1 else 0.0
        features['single_word'] = single_word_score
        
        return features
    
    def _extract_topic_features(self, text: str, words: List[str]) -> Dict[str, float]:
        """Extract features for topic classification"""
        features = {}
        
        # Gaming/casual chat detection
        game_score = 0.0
        for keyword in self.game_keywords:
            if keyword in text:
                game_score = max(game_score, 0.7)
        
        # Boost for exclamation density
        exclamations = text.count('!') + text.count('?') * 0.5
        if exclamations > 0:
            game_score = min(1.0, game_score + exclamations * 0.2)
        
        features['gaming_chat'] = game_score
        
        # Technical/help request detection (inverse of gaming)
        tech_patterns = ["how to", "help with", "error", "issue", "problem", "bug", "fix", "install", "setup"]
        tech_score = 0.0
        for pattern in tech_patterns:
            if pattern in text:
                tech_score = max(tech_score, 0.8)
        features['tech_request'] = tech_score
        
        return features

class ConversationRouter:
    """Main conversation routing engine"""
    
    def __init__(self, embedding_func=None, main_realtime_client=None):
        self.feature_extractor = FeatureExtractor()
        self.threads: List[Thread] = []
        self.embedding_func = embedding_func or self._dummy_embedding
        self.main_realtime_client = main_realtime_client
        self.active_speakers: Set[str] = set()
        
        # Routing thresholds
        self.addr_threshold = 0.55
        self.followup_threshold = 0.45
        self.margin_threshold = 0.2
        
        # Scoring weights (can be tuned)
        self.weights = {
            'addressing': {
                'bot_alias': 1.0,
                'at_mention': 0.8,
                'you_targeting': 0.6,
                'agent_verbs': 0.9,
                'human_address': -0.8  # negative weight - reduces bot addressing
            },
            'followup': {
                'topic_similarity': 0.5,
                'discourse_markers': 0.2,
                'recency_decay': 0.2,
                'participant_bonus': 0.1,
                'expects_reply_bonus': 0.25,
                'question_continuation': 0.15
            },
            'human_to_human': {
                'human_address': 1.5,        # Boost human addressing
                'human_address_pattern': 0.8,
                'bot_alias': -1.0,           # Strong penalty for bot mentions
                'agent_verbs': -0.3          # Slight penalty for imperative verbs
            },
            'game_chatter': {
                'gaming_chat': 1.2,     # Boost gaming detection  
                'confirmation': -0.3,   # confirmations are less likely to be game chatter
                'tech_request': -0.8,   # technical requests are not game chatter
                'human_address': 0.5    # addressing humans can be part of game chatter
            }
        }
    
    def _dummy_embedding(self, text: str) -> List[float]:
        """Dummy embedding function - replace with actual embeddings"""
        # Simple hash-based embedding for demo
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val % (2**31))
        return np.random.randn(384)  # 384-dimensional embedding
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    def decay_function(self, elapsed_seconds: float, half_life: float = 30.0) -> float:
        """Exponential decay function for recency"""
        return np.exp(-elapsed_seconds * np.log(2) / half_life)
    
    def score_addressing_bot(self, features: Dict[str, float]) -> float:
        """Score how likely this turn is addressing the bot"""
        score = 0.0
        weights = self.weights['addressing']
        
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
        
        # Apply sigmoid to get 0-1 range
        return 1 / (1 + np.exp(-score))
    
    def score_followup(self, features: Dict[str, float], thread: Thread, turn: Turn) -> float:
        """Score how likely this turn is following up on a thread"""
        score = 0.0
        weights = self.weights['followup']
        now = time.time()
        
        # Topic similarity
        if turn.embedding is not None:
            base_vec = thread.last_bot_vec if thread.last_bot_vec is not None else thread.topic_vec
            similarity = self.cosine_similarity(turn.embedding, base_vec)
            score += similarity * weights['topic_similarity']
        
        # Discourse markers
        if 'discourse_markers' in features:
            score += features['discourse_markers'] * weights['discourse_markers']
        
        # Question continuation
        if 'question_continuation' in features:
            score += features['question_continuation'] * weights['question_continuation']
        
        # Recency decay
        elapsed = now - thread.last_activity
        recency = self.decay_function(elapsed)
        score += recency * weights['recency_decay']
        
        # Participant bonus - but be very restrictive for single-word responses
        if turn.user_id in thread.participants:
            score += weights['participant_bonus']
        else:
            # Different user - heavily penalize casual single-word responses
            words = turn.text.strip().lower().split()
            if len(words) <= 2 and any(word in ['yeah', 'yes', 'ok', 'okay', 'sure', 'yep', 'no', 'nope'] for word in words):
                score -= 0.5  # Heavy penalty for casual responses from different users
        
        # Expects reply bonus
        if thread.expects_reply and now < thread.expects_reply.get('until', 0):
            score += weights['expects_reply_bonus']
            # Extra bonus for expected response types
            expected_type = thread.expects_reply.get('type')
            if expected_type == 'yesno' and features.get('confirmation', 0) > 0:
                score += 0.3
            elif expected_type == 'choice' and features.get('single_word', 0) > 0:
                score += 0.2
        
        return min(1.0, max(0.0, score))
    
    def score_human_to_human(self, features: Dict[str, float]) -> float:
        """Score how likely this is human-to-human conversation"""
        score = 0.0
        weights = self.weights['human_to_human']
        
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return 1 / (1 + np.exp(-score))
    
    def score_game_chatter(self, features: Dict[str, float]) -> float:
        """Score how likely this is casual gaming chatter"""
        score = 0.0
        weights = self.weights['game_chatter']
        
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return 1 / (1 + np.exp(-score))
    
    def route_turn(self, turn: Turn) -> Dict[str, Any]:
        """Main routing function - decides what to do with a turn"""
        # Skip very low confidence or empty turns
        if turn.confidence < 0.55 or len(turn.text.strip()) < 1:
            return {"action": "ignore", "reason": "low_confidence"}
        
        # Track active speakers
        self.active_speakers.add(turn.display_name)
        
        # Extract features with user context  
        features = self.feature_extractor.extract_features(turn, list(self.active_speakers))
        
        # Generate embedding if not provided
        if turn.embedding is None:
            turn.embedding = self.embedding_func(turn.text)
        
        now = time.time()
        
        # Score all routing options
        scores = {}
        
        # Direct bot addressing
        scores['addr_bot'] = self.score_addressing_bot(features)
        
        # Follow-up scores for each active thread
        followup_scores = []
        best_followup_score = 0.0
        for i, thread in enumerate(self.threads):
            followup_score = self.score_followup(features, thread, turn)
            followup_scores.append((i, followup_score))
            scores[f'followup_{i}'] = followup_score
            best_followup_score = max(best_followup_score, followup_score)
        
        # If no threads exist, don't consider follow-ups
        if not self.threads:
            best_followup_score = 0.0
        
        # Human-to-human conversation
        scores['human_to_human'] = self.score_human_to_human(features)
        
        # Game chatter
        scores['game_chatter'] = self.score_game_chatter(features)
        
        # Find the best scoring option (exclude followup options if no threads)
        if not self.threads:
            # Filter out followup keys if no threads exist
            valid_scores = {k: v for k, v in scores.items() if not k.startswith('followup_')}
        else:
            valid_scores = scores
        
        best_key = max(valid_scores.keys(), key=lambda k: valid_scores[k])
        best_score = valid_scores[best_key]
        
        # Decision logic with priority for strong bot addressing
        # If there's a clear bot alias mention, prioritize direct addressing over followup
        if (best_key == 'addr_bot' and best_score > self.addr_threshold) or \
           (features.get('bot_alias', 0) >= 1.0 and scores.get('addr_bot', 0) > 0.7):
            # Start new thread
            thread = self._spawn_thread(turn)
            return {
                "action": "speak",
                "thread_id": thread.id,
                "reason": "direct_addressing",
                "score": best_score,
                "features": features
            }
        
        elif best_key.startswith('followup_') and best_score > self.followup_threshold:
            # Continue existing thread
            thread_idx = int(best_key.split('_')[1])
            thread = self.threads[thread_idx]
            thread.add_participant(turn.user_id)
            thread.update_activity()
            return {
                "action": "speak", 
                "thread_id": thread.id,
                "reason": "followup",
                "score": best_score,
                "features": features
            }
        
        # Check if human/game clearly dominates
        second_best_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
        margin = best_score - second_best_score
        
        if (best_key in ['human_to_human', 'game_chatter'] and 
            best_score > 0.3 and margin > self.margin_threshold):  # Lower threshold for human detection
            # Ignore - clearly not for the bot
            self._maybe_close_stale_threads()
            return {
                "action": "ignore",
                "reason": best_key,
                "score": best_score,
                "margin": margin,
                "features": features
            }
        
        # Special case: If there's strong human addressing without bot mentions, ignore
        if (features.get('human_address', 0) > 0.8 and 
            features.get('bot_alias', 0) < 0.1 and
            features.get('agent_verbs', 0) < 0.5):
            return {
                "action": "ignore", 
                "reason": "strong_human_addressing",
                "score": features.get('human_address', 0),
                "features": features
            }
        
        # Default to ignore if no clear winner
        return {
            "action": "ignore",
            "reason": "no_clear_winner", 
            "best_score": best_score,
            "features": features
        }
    
    def _spawn_thread(self, turn: Turn) -> Thread:
        """Create a new conversation thread"""
        thread_id = f"thread_{int(time.time())}_{turn.user_id[:8]}"
        thread = Thread(
            id=thread_id,
            topic_vec=turn.embedding[:] if turn.embedding else [],
            participants={turn.user_id}
        )
        self.threads.append(thread)
        log.info(f"Spawned new thread: {thread_id}")
        return thread
    
    def on_bot_reply(self, thread_id: str, reply_text: str, question_type: Optional[str] = None):
        """Handle bot reply - update thread state and set expectations"""
        thread = self._get_thread_by_id(thread_id)
        if not thread:
            return
        
        thread.update_activity()
        thread.last_bot_vec = self.embedding_func(reply_text)
        
        # Set reply expectations if this is a question
        if question_type:
            thread.expects_reply = {
                'type': question_type,  # 'yesno', 'choice', 'freeform'
                'until': time.time() + 8.0  # 8 second window
            }
            log.debug(f"Thread {thread_id} expects {question_type} reply")
        
        # Clear expectations after timeout
        if thread.expects_reply:
            def clear_expectations():
                if thread.expects_reply and time.time() > thread.expects_reply['until']:
                    thread.expects_reply = None
            
            # Schedule expectation clearing (only if event loop is running)
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(asyncio.sleep(9.0))
                task.add_done_callback(lambda _: clear_expectations())
            except RuntimeError:
                # No running loop - skip expectation clearing in test mode
                pass
    
    def _get_thread_by_id(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        for thread in self.threads:
            if thread.id == thread_id:
                return thread
        return None
    
    def _maybe_close_stale_threads(self, max_age_seconds: float = 60.0):
        """Close threads that haven't been active recently"""
        now = time.time()
        active_threads = []
        
        for thread in self.threads:
            if (now - thread.last_activity) < max_age_seconds:
                active_threads.append(thread)
            else:
                log.info(f"Closed stale thread: {thread.id}")
        
        self.threads = active_threads
    
    def classify_bot_question(self, text: str) -> Optional[str]:
        """Classify if bot's response contains a question and what type"""
        text = text.lower().strip()
        
        if not text.endswith('?'):
            return None
        
        # Yes/no questions
        yesno_patterns = [
            r'\b(do|does|did|is|are|was|were|will|would|can|could|should|shall)\s',
            r'\b(right|correct|true|ok|okay)\?',
            r'\bor no\?'
        ]
        
        for pattern in yesno_patterns:
            if re.search(pattern, text):
                return 'yesno'
        
        # Choice questions
        if ' or ' in text and text.count('?') == 1:
            return 'choice'
        
        # Default to freeform
        return 'freeform'
    
    def get_active_threads(self) -> List[Dict[str, Any]]:
        """Get info about currently active threads"""
        now = time.time()
        thread_info = []
        
        for thread in self.threads:
            info = {
                'id': thread.id,
                'participants': list(thread.participants),
                'age_seconds': now - thread.last_activity,
                'expects_reply': bool(thread.expects_reply),
                'reply_type': thread.expects_reply.get('type') if thread.expects_reply else None,
                'reply_expires_in': max(0, thread.expects_reply.get('until', 0) - now) if thread.expects_reply else 0
            }
            thread_info.append(info)
        
        return thread_info
    
    def update_thresholds(self, addr_threshold: float = None, followup_threshold: float = None, margin_threshold: float = None):
        """Update routing thresholds for tuning"""
        if addr_threshold is not None:
            self.addr_threshold = addr_threshold
        if followup_threshold is not None:
            self.followup_threshold = followup_threshold  
        if margin_threshold is not None:
            self.margin_threshold = margin_threshold
        
        log.info(f"Updated thresholds - addr: {self.addr_threshold}, followup: {self.followup_threshold}, margin: {self.margin_threshold}")


def argmax(scores: Dict[str, float]) -> Dict[str, Any]:
    """Find the key with maximum score"""
    if not scores:
        return {'key': None, 'val': 0}
    
    best_key = max(scores.keys(), key=lambda k: scores[k])
    return {'key': best_key, 'val': scores[best_key]}