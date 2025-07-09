#!/usr/bin/env python3

import json
import logging
import os
from transformers import AutoConfig

class FlopCounter():
    def __init__(self, modelpath="../../models/gpt2"):
        self.modelpath = modelpath
        self.params = self.get_params()

    def parse_config(self, config, key):
        synonyms = [["n_inner", "ff_intermediate_size", "intermediate_size", "d_ff"],
                    ["hidden_act", "activation_function"],
                    ["n_embd", "hidden_size", "d_model"],
                    ["n_head", "num_heads", "num_attention_heads"],
                    ["n_layer", "num_layers", "num_hidden_layers"],
                    ["is_encoder", "is_decoder", "is_encoder_decoder"]]
        
        keys_to_test = [key]
        for s in synonyms[:-1]:
            if key in s:
                keys_to_test = s

        # handle special case of detecting enc/dec/enc-dec
        if key in synonyms[-1]:
            for s in synonyms[-1]:
                try:
                    if config[s]:
                        return s[3:]
                except KeyError:
                    continue
            return None

        out = None
        for k in keys_to_test:
            try:
                out = config[k]
                
                # Convert any strings to lowercase and expect that to fail for ints or other
                try:
                    out.lower()
                except AttributeError:
                    pass
                
                # Exit the look on first match
                if out is not None:
                    break

            # Expect the config to not specify the key or, more likely, its synonym
            except KeyError:
                continue

        return out
    
    def infer_params(self, params):
        essential_params = ["d_embed", "n_head", "n_vocab", "n_layer"]
        synonyms = [["geglu", "gegelu"]]

        for e in essential_params:
            if params[e] is None:
                raise ValueError(f"Cannot determine computational complexity: parameter {e} unspecified!")

        # Activation needs to be fixed before d_ffn
        if params["activation"] is not None:
            for s in synonyms:
                if params["activation"] in s:
                    params["activation"] = s[0]
        else:
            params["activation"] = "relu"
            logging.warning(f"Complexity: Assuming activation = {params["activation"]}")
            
        
        if params["d_ffn"] is None:
            if "glu" in params["activation"]:
                # Assume T5-Style GLU with D x 8D/3 aspect ratio
                params["d_ffn"] = int(params["d_embed"] * 8/3)
            else:
                # Assume conventional D x 4D MLP
                params["d_ffn"] = params["d_embed"] * 4

            logging.warning(f"Complexity: Assuming d_ffn = {params["d_ffn"]}")

        for layer in ["ln", "attn", "mlp"]:
            if params[f"{layer}_bias"] is None:
                # Prefer to underestimate complexity
                params[f"{layer}_bias"] = False
                logging.warning(f"Complexity: Assuming {layer}_bias = {params[f"{layer}_bias"]}")
        
        if params["n_kvhead"] is None:
            params["n_kvhead"] = params["n_head"]
            logging.warning(f"Complexity: Assuming n_kvhead = {params["n_kvhead"]} (No MQA/GQA)")
        
        if params["type"] is None:
            params["type"] = "decoder"
            logging.warning(f"Complexity: Assuming type = {params["type"]}")
        
        return params
    
    def get_params_for_known_models(self, modelpath):
        known_params = {"bert-base-uncased": {
                            "d_embed": 768, "d_ffn": 3072, "ln_bias": True, "attn_bias": True, "mlp_bias": True,
                            "activation": "gelu", "n_head": 12, "n_kvhead": 12, "n_vocab": 30522, "n_layer": 12,
                            "kvcache": False, "type": "encoder"},
                        "gpt2": {
                            "d_embed": 768, "d_ffn": 3072, "ln_bias": True, "attn_bias": True, "mlp_bias": True,
                            "activation": "gelu", "n_head": 12, "n_kvhead": 12, "n_vocab": 50257, "n_layer": 12,
                            "kvcache": True, "type": "decoder"},
                        "phi-3-small-8k-instruct": {
                            "d_embed": 4096, "d_ffn": 14336, "ln_bias": True, "attn_bias": True, "mlp_bias": True,
                            "activation": "geglu", "n_head": 32, "n_kvhead": 8, "n_vocab": 100352, "n_layer": 32,
                            "kvcache": True, "type": "decoder"},
                        "t5-small": {
                            "d_embed": 512, "d_ffn": 2048, "ln_bias": True, "attn_bias": False, "mlp_bias": False,
                            "activation": "relu", "n_head": 8, "n_kvhead": 8, "n_vocab": 32128, "n_layer": 12,
                            "kvcache": True, "type": "encoder-decoder"},
                        "teuken-7b-instruct-research-v0.4": {
                            "d_embed": 4096, "d_ffn": 13440, "ln_bias": True, "attn_bias": True, "mlp_bias": True,
                            "activation": "swiglu", "n_head": 32, "n_kvhead": 2, "n_vocab": 250680, "n_layer": 32,
                            "kvcache": True, "type": "decoder"}}

        modelpath = modelpath.split("/")[-1].lower()
        if modelpath in known_params.keys():
            return True, known_params[modelpath]
        else:
            print(f"Model {modelpath} not known :/")
            return False, {}

    def get_params(self):
        # 1. See if the model is known and if so, return its parameters.
        # 2. If not, then load the model config and parse as many parameters as possible
        # 3. Attempt to infer missing data from parsed fields
        
        # Step 1
        got_updated, params = self.get_params_for_known_models(self.modelpath)
        if got_updated:
            return params
        
        # Step 2
        params = {
            # Dimensions
            "d_embed": None, # cared for
            "d_ffn": None, # cared for

            # Whether to account for Biases
            "ln_bias": None, # cared for
            "attn_bias": None, # cared for
            "mlp_bias": None, # cared for

            # Variants, Other parameters
            "activation": None, # this would also encode GLU usage # cared for
            "n_head": None, # cared for
            "n_kvhead": None,
            "n_vocab": None, # cared for
            "n_layer": None, # cared for
            "kvcache": None,
            "type": None # cared for
        }

        config = AutoConfig.from_pretrained(self.modelpath, trust_remote_code=True).to_dict()

        params["d_embed"] = self.parse_config(config, "n_embd")
        params["d_ffn"] = self.parse_config(config, "n_inner")

        params["attn_bias"] = self.parse_config(config, "attention_bias")

        params["activation"] = self.parse_config(config, "activation_function")
        params["n_head"] = self.parse_config(config, "n_head")
        params["n_kvhead"] = self.parse_config(config, "num_key_value_heads")
        params["n_vocab"] = self.parse_config(config, "vocab_size")
        params["n_layer"] = self.parse_config(config, "n_layer")
        params["kvcache"] = self.parse_config(config, "use_cache")
        params["type"] = self.parse_config(config, "is_decoder") # ???
        
        # Step 3
        params = self.infer_params(params)
        return params
    
    def _calc_embed_cpx(self, D):
        ...
    
    def _calc_residual()
    
    def calc_complexity(self, params):
        in_tokens = 100
        gen_tokens = 100
        
        if params["kvcache"]:
            prefill = _calc_prefill()
            kvcached = _calc_kvcached()
        else:
            iteration 
            _
        attn = _calc_attn_cpx()
        

if __name__ == "__main__":
    print("Phi")
    fc = FlopCounter("../../models/Phi-3-small-8k-instruct")
    print("GPT2")
    fc = FlopCounter("../../models/gpt2")
    print("BERT")
    fc = FlopCounter("../../models/bert-base-uncased")
    print("T5")
    fc = FlopCounter("../../models/t5-small")
    print("Teuken")
    fc = FlopCounter("../../models/Teuken-7B-instruct-research-v0.4")
