#!/bin/bash
source env.sh

pip install torch==2.6.0

pip install --no-build-isolation accelerate==1.6.0 transformers==4.50.3 sentencepiece==0.2.0

pip install --no-build-isolation vllm==0.8.2

pip install --no-build-isolation llmlingua==0.2.2

pip install --no-build-isolation flash-attn==2.7.4.post1

# pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

pip install --no-build-isolation deepspeed==0.16.5 deepspeed-mii==0.3.3

echo "Setup complete!"