from transformers import pipeline
import time

print("===== Part 2: Default Sentiment Pipeline =====")
start_time = time.time()

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face models!")

end_time = time.time()

print("Result:", result)
print("Time taken:", end_time - start_time, "seconds")
print()

print("===== Part 3: Finance Model =====")
financial_classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

text = "The stock market rally continued, suggesting strong long-term growth."
finance_result = financial_classifier(text)

print("Finance Result:", finance_result)
print()

print("===== Part 4: Bulk Analysis =====")
sentences = [
    "The quarterly earnings report was surprisingly weak, causing investor concern.",
    "Despite market volatility, the company announced record profits.",
    "I'm not sure if I should invest in tech stocks this quarter."
]

bulk_results = financial_classifier(sentences)

for i, item in enumerate(bulk_results, start=1):
    print(f"Sentence {i} Result:", item)
print()

print("===== Part 5: Zero-Shot Classification =====")
model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text2 = "The latest press release details the company's new policy on remote work, including guidelines for team communication and hardware allocation for employees worldwide."

labels1 = ["Employee Relations", "Financial News", "Product Announcement", "Technical Support"]

result1 = model(text2, candidate_labels=labels1)

print("Original labels result:")
print(result1)
print()

print("Detailed ranking:")
for label, score in zip(result1["labels"], result1["scores"]):
    print(label, ":", score)
print()

labels2 = [
    "Employee Relations",
    "Financial News",
    "Product Announcement",
    "Technical Support",
    "Sales",
    "HR Policy",
    "Legal Compliance"
]

result2 = model(text2, candidate_labels=labels2)

print("Extended labels result:")
print(result2)
print()

print("Detailed ranking with extended labels:")
for label, score in zip(result2["labels"], result2["scores"]):
    print(label, ":", score)