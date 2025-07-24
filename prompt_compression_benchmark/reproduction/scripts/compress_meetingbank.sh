# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ======================= Meetingbank =======================
python compress.py --load_origin_from ../results/meetingbank_short/origin/meetingbank_test_formated.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --compression_rate 0.33 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/meetingbank_short/llmlingua2/compression_ratio33_meetingbank_test_formated.json
python compress.py --load_origin_from ../results/meetingbank_short/origin/meetingbank_test_formated.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --compression_rate 0.33 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/meetingbank_short/llmlingua2_small/compression_ratio33_meetingbank_test_formated.json
