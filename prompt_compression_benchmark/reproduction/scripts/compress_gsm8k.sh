# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ======================= GSM8K =======================
python compress.py --load_origin_from ../results/gsm8k/origin/gsm8k_cot_example.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:7 \
    --load_key prompt_list \
    --target_token 400 \
    --force_tokens "+,-,*,×,/,÷,=,The answer is,\n" \
    --use_context_level_filter \
    --force_reserve_digit \
    --save_path ../results/gsm8k/llmlingua2/compression_target400_gsm8k_cot_example.json
python compress.py --load_origin_from ../results/gsm8k/origin/gsm8k_cot_example.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:7 \
    --load_key prompt_list \
    --target_token 160 \
    --force_tokens "+,-,*,×,/,÷,=,The answer is,\n" \
    --use_context_level_filter \
    --force_reserve_digit \
    --save_path ../results/gsm8k/llmlingua2/compression_target160_gsm8k_cot_example.json
python compress.py --load_origin_from ../results/gsm8k/origin/gsm8k_cot_example.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:7 \
    --load_key prompt_list \
    --target_token 400 \
    --force_tokens "+,-,*,×,/,÷,=,The answer is,\n" \
    --use_context_level_filter \
    --force_reserve_digit \
    --save_path ../results/gsm8k/llmlingua2_small/compression_target400_gsm8k_cot_example.json
python compress.py --load_origin_from ../results/gsm8k/origin/gsm8k_cot_example.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:7 \
    --load_key prompt_list \
    --target_token 160 \
    --force_tokens "+,-,*,×,/,÷,=,The answer is,\n" \
    --use_context_level_filter \
    --force_reserve_digit \
    --save_path ../results/gsm8k/llmlingua2_small/compression_target160_gsm8k_cot_example.json
