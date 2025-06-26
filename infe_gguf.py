from vllm import LLM, SamplingParams

conversation = [
   {
      "role": "system",
      "content": "Do you know Japan?"
   }
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
llm = LLM(
        model="/home/z890/model/tinyllama-1.1B-chat/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        tokenizer="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tensor_parallel_size=2
    )
outputs = llm.chat(conversation, sampling_params)

for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")