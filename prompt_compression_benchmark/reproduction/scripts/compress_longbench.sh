# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# ======================= Longbench =======================
python compress.py --load_origin_from ../results/longbench/origin/longbench_test_single_doc_qa_formated.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:6 \
    --load_key context \
    --target_token 2000 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/longbench/llmlingua2/compression_target2000_longbench_test_single_doc_qa_formated.json
python compress.py --load_origin_from ../results/longbench/origin/longbench_test_single_doc_qa_formated.json \
    --model_name microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    --device cuda:7 \
    --load_key context \
    --target_token 3000 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/longbench/llmlingua2/compression_target3000_longbench_test_single_doc_qa_formated.json
python compress.py --load_origin_from ../results/longbench/origin/longbench_test_single_doc_qa_formated.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:5 \
    --load_key context \
    --target_token 2000 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/longbench/llmlingua2_small/compression_target2000_longbench_test_single_doc_qa_formated.json
python compress.py --load_origin_from ../results/longbench/origin/longbench_test_single_doc_qa_formated.json \
    --model_name microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --device cuda:5 \
    --load_key context \
    --target_token 3000 \
    --force_tokens "\n,?,!,." \
    --save_path ../results/longbench/llmlingua2_small/compression_target3000_longbench_test_single_doc_qa_formated.json
