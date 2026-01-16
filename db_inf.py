from transformers import pipeline

# Load DistilBERT masked LM
fill_mask = pipeline("fill-mask", model="distilbert-base-uncased")

prompt = "A dog is a [MASK] animal."


results = fill_mask(prompt)
print(f"\n{prompt}")
for result in results[:3]:  # Top 3
    print(f"  {result['token_str']}: {result['score']:.2%}")