from llama_cpp import Llama

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path="/home/roxasrr/source/models/Lexi-Llama-3-8B-Uncensored-Q8_0.gguf",  # Download the model file first
  n_ctx=2048,  # The max sequence length to use - note that longer sequence lengths require much more resources
  n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
  n_gpu_layers=2          # The number of layers to offload to GPU, if you have GPU acceleration available
)

# Simple inference example
message = " News: (Dec 22, 2013  3:00 PM) You probably felt pretty old when you learned Brad Pitt had turned 50, prompting AARP to invite him to join their ranks. Well, Merry Christmas, we're about to make you feel even older: The San Francisco Chronicle rounds up 14 more celebrities who hit 50 this year. Click through the gallery for a sampling, or check out the complete list here.\nHeadline: 8 Stars Who Hit ____ This Year\n\nWhat is the value of ___? Only give a numerical Response: "
prompt = f"[INST] {message} [/INST]"
output = llm(
  prompt, # Prompt
  max_tokens=512,  # Generate up to 512 tokens
  stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
  echo=True        # Whether to echo the prompt
)
print(output)