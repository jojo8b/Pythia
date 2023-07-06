from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
"<path-to-model>/pythia-6.9b-deduped"
)

tokenizer = AutoTokenizer.from_pretrained(
"<path-to-model>/pythia-6.9b-deduped"
)

inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])

inputs = tokenizer("How many books are there in the Bible", return_tensors="pt")
tokens = model.generate(**inputs)
tokenizer.decode(tokens[0])

# original example 
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
  revision="step3000",
  cache_dir="../code/pythia-6.9b-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-6.9b-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

