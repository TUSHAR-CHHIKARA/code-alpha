import spacy
from spacy.matcher import PhraseMatcher

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

# Frequently Asked Questions (FAQs)
faq = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase with a receipt.",
    "How can I track my order?": "You can track your order using the tracking number provided in your confirmation email.",
    "Do you offer international shipping?": "Yes, we offer international shipping to many countries.",
    "How can I contact customer support?": "You can contact customer support via email at support@example.com or call us at 1-800-123-4567.",
    "What payment methods do you accept?": "We accept all major credit cards, PayPal, and Apple Pay.",
}

# Preprocess FAQs for matching
patterns = list(faq.keys())
pattern_docs = list(nlp.pipe(patterns))
matcher = PhraseMatcher(nlp.vocab)
for pattern_doc in pattern_docs:
    matcher.add(pattern_doc.text, [pattern_doc])

# Function to find the best matching FAQ
def find_faq(question):
    doc = nlp(question)
    matches = matcher(doc)
    if matches:
        # Find the best match (the one with the longest span)
        best_match = max(matches, key=lambda match: match[2] - match[1])
        matched_span = doc[best_match[1]:best_match[2]]
        return faq.get(matched_span.text, "Sorry, I don't have an answer for that question.")
    else:
        return "Sorry, I don't have an answer for that question."

# Chatbot loop
print("Welcome to the FAQ chatbot! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    response = find_faq(user_input)
    print(f"Bot: {response}")
