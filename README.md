# 1️⃣ AI Chatbot (Rule-based + NLP)

## 1) Project Title
**Simple Intent Chatbot (Rule-based + NLP)**

## 2) Problem Statement
Build a lightweight chatbot that can respond to common user messages (greetings, help, weather, time, goodbye, thanks) without using a heavy neural model. The goal is to combine clear rule-based responses with simple NLP intent matching for robust behavior on short text inputs.

## 3) AI Concept Used
- **Rule-based AI**: predefined intents and responses.
- **NLP intent matching**: TF-IDF vectorization + cosine similarity to map user text to the nearest intent example.
- **Confidence thresholding**: avoids incorrect confident answers by using a fallback response.

## 4) Dataset / Input Description
This project uses an **inline mini intent dataset** (no external download required):
- Intents: `greeting`, `goodbye`, `thanks`, `help`, `weather`, `time`.
- Each intent has:
  - example phrases (training text)
  - multiple response templates

Input:
- A single user message in natural language via terminal.

Output:
- A chatbot response based on best-matching intent or fallback.

## 5) Full Python Code
```python
"""Simple Rule-based + NLP chatbot.

Python 3.10+
Dependencies:
    - scikit-learn
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class IntentData:
    examples: List[str]
    responses: List[str]


class IntentChatbot:
    """A compact chatbot that mixes rule-based logic with NLP intent matching."""

    def __init__(self, confidence_threshold: float = 0.25) -> None:
        self.confidence_threshold = confidence_threshold
        self.fallback_responses = [
            "I didn't fully understand that. Could you rephrase?",
            "I'm not sure about that yet. Try asking in a different way.",
            "Sorry, I missed that. Can you be a bit more specific?",
        ]

        self.intent_map: Dict[str, IntentData] = {
            "greeting": IntentData(
                examples=[
                    "hi",
                    "hello",
                    "hey",
                    "good morning",
                    "good evening",
                    "how are you",
                ],
                responses=[
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! Ask me anything about this demo chatbot.",
                ],
            ),
            "goodbye": IntentData(
                examples=[
                    "bye",
                    "goodbye",
                    "see you later",
                    "talk to you soon",
                    "i have to go",
                ],
                responses=[
                    "Goodbye! Have a great day.",
                    "See you soon!",
                    "Take care!",
                ],
            ),
            "thanks": IntentData(
                examples=[
                    "thanks",
                    "thank you",
                    "much appreciated",
                    "thanks a lot",
                ],
                responses=[
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime!",
                ],
            ),
            "help": IntentData(
                examples=[
                    "can you help me",
                    "i need help",
                    "what can you do",
                    "how does this work",
                    "what are your features",
                ],
                responses=[
                    "I can handle greetings, thanks, weather-style questions, time requests, and simple help.",
                    "Try asking me: 'what time is it?' or 'tell me the weather'.",
                ],
            ),
            "weather": IntentData(
                examples=[
                    "what is the weather",
                    "weather today",
                    "is it raining",
                    "forecast",
                    "how is the weather outside",
                ],
                responses=[
                    "I don't have live weather APIs connected, but it might be a good idea to check a weather app.",
                    "I can't fetch real-time weather yet, but I can still help with other questions.",
                ],
            ),
            "time": IntentData(
                examples=[
                    "what time is it",
                    "current time",
                    "tell me the time",
                    "can you give me the time",
                ],
                responses=[],
            ),
        }

        self._fit_vectorizer()

    def _fit_vectorizer(self) -> None:
        """Prepare TF-IDF vectors from intent examples."""
        self.example_texts: List[str] = []
        self.example_labels: List[str] = []

        for intent_name, data in self.intent_map.items():
            for example in data.examples:
                self.example_texts.append(self._clean_text(example))
                self.example_labels.append(intent_name)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.example_vectors = self.vectorizer.fit_transform(self.example_texts)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Lowercase and remove extra symbols for stable matching."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _predict_intent(self, user_text: str) -> Tuple[str, float]:
        """Return best intent and similarity score."""
        cleaned = self._clean_text(user_text)
        query_vec = self.vectorizer.transform([cleaned])
        sims = cosine_similarity(query_vec, self.example_vectors)[0]
        best_idx = sims.argmax()
        return self.example_labels[best_idx], float(sims[best_idx])

    def get_response(self, user_text: str) -> str:
        """Generate chatbot reply from user text."""
        intent, score = self._predict_intent(user_text)

        if score < self.confidence_threshold:
            return random.choice(self.fallback_responses)

        if intent == "time":
            now = datetime.now().strftime("%H:%M:%S")
            return f"Current local time is {now}."

        responses = self.intent_map[intent].responses
        return random.choice(responses) if responses else "Okay."


def run_chat() -> None:
    """Interactive command-line chat loop."""
    print("Simple Intent Chatbot")
    print("Type 'quit' to exit.\n")

    bot = IntentChatbot(confidence_threshold=0.25)

    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            print("Bot: Goodbye! 👋")
            break

        response = bot.get_response(user)
        print(f"Bot: {response}")


if __name__ == "__main__":
    run_chat()
```

## 6) Explanation of Workflow
1. **Create intent knowledge base**
   - Define intents with example phrases and responses.
2. **Preprocess text**
   - Lowercase, remove punctuation, normalize spaces.
3. **Vectorize examples**
   - Convert text to TF-IDF vectors using unigrams + bigrams.
4. **Match user input to an intent**
   - Transform user text to vector.
   - Compute cosine similarity against all example vectors.
   - Select highest score intent.
5. **Apply rule-based response policy**
   - If score is below threshold → fallback message.
   - If intent is `time` → dynamic response with current time.
   - Else → random canned response from selected intent.

## 7) How to Run Instructions
1. Make sure Python 3.10+ is installed.
2. Install dependency:
   ```bash
   pip install scikit-learn
   ```
3. Save code as `chatbot.py`.
4. Run:
   ```bash
   python chatbot.py
   ```
5. Chat in terminal and type `quit` to stop.
