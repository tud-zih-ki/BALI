# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ======================= BBH =======================
python compress.py --load_origin_from ../results/bbh/origin/bbh_cot_examples.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:5 \
    --load_key prompt_list \
    --target_token 300 \
    --force_tokens "\n,!,?,.,Q:,A:,So the answer is" \
    --use_context_level_filter \
    --save_path ../results/bbh/llmlingua2/compression_target300_bbh_cot_examples.json
python compress.py --load_origin_from ../results/bbh/origin/bbh_cot_examples.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:5 \
    --load_key prompt_list \
    --target_token 200 \
    --force_tokens "\n,!,?,.,Q:,A:,So the answer is" \
    --use_context_level_filter \
    --save_path ../results/bbh/llmlingua2/compression_target200_bbh_cot_examples.json
python compress.py --load_origin_from ../results/bbh/origin/bbh_cot_examples.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:5 \
    --load_key prompt_list \
    --target_token 300 \
    --force_tokens "\n,!,?,.,Q:,A:,So the answer is" \
    --use_context_level_filter \
    --save_path ../results/bbh/llmlingua2_small/compression_target300_bbh_cot_examples.json
python compress.py --load_origin_from ../results/bbh/origin/bbh_cot_examples.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:5 \
    --load_key prompt_list \
    --target_token 200 \
    --force_tokens "\n,!,?,.,Q:,A:,So the answer is" \
    --use_context_level_filter \
    --save_path ../results/bbh/llmlingua2_small/compression_target200_bbh_cot_examples.json
