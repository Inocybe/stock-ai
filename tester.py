from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-4B-8bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)

print(text)