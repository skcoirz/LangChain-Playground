
def example_with_trust_remote_code():
	import torch
	from transformers import pipeline
	generate_text = pipeline(
		model="databricks/dolly-v2-3b",
		torch_dtype=torch.bfloat16,
		trust_remote_code=True,
		device_map="auto"
	)

	res = generate_text("Explain to me the difference between nuclear fission and fusion.")
	print(res[0]["generated_text"])

# example_with_trust_remote_code()

def example_with_everything_local():
	import torch
	from transformers import pipeline
	from transformers import AutoModelForCausalLM, AutoTokenizer

	tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
	model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", device_map="auto", torch_dtype=torch.float16)
	pipeline = pipeline(
    	task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device="mps",
        # model_kwargs={},
    )
	res = generate_text("Explain to me the difference between nuclear fission and fusion.")
	print(res[0]["generated_text"])

example_with_everything_local()
# this takes forever...

