# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ======================= Zero Scrolls =======================
python compress.py --load_origin_from ../results/zero_scrolls/origin/zero_scrolls_validation.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:7 \
    --target_token 2000 \
    --force_tokens '\n,?,!,.' \
    --save_path ../results/zero_scrolls/llmlingua2/compression_target2000_zero_scrolls_validation.json
python compress.py --load_origin_from ../results/zero_scrolls/origin/zero_scrolls_validation.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:0 \
    --target_token 3000 \
    --force_tokens '\n,?,!,.' \
    --save_path ../results/zero_scrolls/llmlingua2/compression_target3000_zero_scrolls_validation.json
python compress.py --load_origin_from ../results/zero_scrolls/origin/zero_scrolls_validation.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:0 \
    --target_token 2000 \
    --force_tokens '\n,?,!,.' \
    --save_path ../results/zero_scrolls/llmlingua2_small/compression_target2000_zero_scrolls_validation.json
python compress.py --load_origin_from ../results/zero_scrolls/origin/zero_scrolls_validation.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:0 \
    --target_token 3000 \
    --force_tokens '\n,?,!,.' \
    --save_path ../results/zero_scrolls/llmlingua2_small/compression_target3000_zero_scrolls_validation.json
