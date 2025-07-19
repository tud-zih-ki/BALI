#!/usr/bin/env python3

import logging
from transformers import AutoConfig

# Complexity of the Residual Stream matrix addidion
def flops_residual(d_embed, d_input):
    return d_embed * d_input

# Complexity of the Layer Norm with optional biases
def flops_layer_norm(d_embed, d_input, ln_bias):
    result = 5 * d_embed * d_input + 2 * d_input
    if ln_bias:
        result += 2 * d_embed * d_input
    return result

# Complexity of a softmax function over an n-dimensional vector
def flops_softmax(n):
    return 3 * n - 1

def flops_attn(d_embed, d_input, n_head, n_kvhead, causal, attn_bias):
    inproj = (2 + 4 * n_kvhead / n_head) * d_embed * d_embed * d_input
    outproj = 2 * d_embed * d_embed * d_input

    if not attn_bias:
        inproj -= (n_head + 2 * n_kvhead) * d_embed / n_head * d_input
        outproj -= d_embed * d_input

    if causal:
        qkt = d_embed * d_input * d_input + d_embed * d_input
        softmax = n_head * (d_input * flops_softmax(d_input)) / 2
        vout = d_embed*d_input*d_input
    else:
        qkt = 2*d_embed*d_input*d_input
        softmax = n_head * (d_input * flops_softmax(d_input))
        vout = 2*d_embed*d_input*d_input - d_embed*d_input

    result = inproj + qkt + softmax + vout + outproj
    return result

# Complexity of the MLP with different activations and GLU
def flops_mlp(d_embed, d_ffn, d_input, act, mlp_bias):
    # dict(activation: (complexity, is_glu))
    activations = {"relu": (1, False), "gelu": (9, False), "swish": (4, False),
                   "reglu": (1, True), "geglu": (9, True), "swiglu": (4, True)}
    if act in activations:
        act_cpx, glu = activations[act]
    else:
        act_cpx, glu = activations["relu"]

    if glu:
        result = 6 * d_ffn * d_embed * d_input + act_cpx * d_ffn * d_input
        if not mlp_bias:
            result -= 3 * d_ffn * d_input
    else:
        result = 4 * d_ffn * d_embed * d_input + act_cpx * d_ffn * d_input
        if not mlp_bias:
            result -= 2 * d_ffn * d_input

    return result

def flops_embedding(d_embed, d_input):
    return d_embed * d_input

def flops_decoding(d_embed, d_input, n_vocab):
    return 2 * d_embed * d_input * n_vocab + 2 * d_input * n_vocab - d_input

def flops_forward(d_input, d_embed, d_ffn, ln_bias, attn_bias, mlp_bias, activation,
                     n_head, n_kvhead, n_vocab, n_layer, kvcache, type):
    if type == "decoder":
        if kvcache:
            result = flops_embedding(d_embed, 1) \
                     + n_layer * (2 * flops_residual(d_embed, 1) \
                                  + 2 * flops_layer_norm(d_embed, 1, ln_bias) \
                                  + flops_attn(d_embed, d_input, n_head, n_kvhead, False, attn_bias) / d_input \
                                  + flops_mlp(d_embed, d_ffn, 1, activation, mlp_bias)) \
                     + flops_decoding(d_embed, 1, n_vocab)
        else:
            result = flops_embedding(d_embed, d_input) \
                     + n_layer * (2 * flops_residual(d_embed, d_input) \
                                  + 2 * flops_layer_norm(d_embed, d_input, ln_bias) \
                                  + flops_attn(d_embed, d_input, n_head, n_kvhead, True, attn_bias) \
                                  + flops_mlp(d_embed, d_ffn, d_input, activation, mlp_bias)) \
                     + flops_decoding(d_embed, d_input, n_vocab)

    elif type == "encoder":
        result = flops_embedding(d_embed, d_input) \
                 + n_layer * (2 * flops_residual(d_embed, d_input) \
                              + 2 * flops_layer_norm(d_embed, d_input, ln_bias) \
                              + flops_attn(d_embed, d_input, n_head, n_kvhead, False, attn_bias) \
                              + flops_mlp(d_embed, d_ffn, d_input, activation, mlp_bias))

    else:
        result = flops_embedding(d_embed, d_input) \
                 + n_layer / 2 * (2 * flops_residual(d_embed, d_input) \
                                  + 2 * flops_layer_norm(d_embed, d_input, ln_bias) \
                                  + flops_attn(d_embed, d_input, n_head, n_kvhead, False, attn_bias)\
                                  + flops_mlp(d_embed, d_ffn, d_input, activation, mlp_bias)) \
                 + n_layer / 2 * (3 * flops_residual(d_embed, d_input) \
                                  + 3 * flops_layer_norm(d_embed, d_input, ln_bias) \
                                  + flops_attn(d_embed, d_input, n_head, n_kvhead, True, attn_bias) \
                                  + flops_attn(d_embed, d_input, n_head, n_kvhead, False, attn_bias) \
                                  + flops_mlp(d_embed, d_ffn, d_input, activation, mlp_bias)) \
                 + flops_decoding(d_embed, d_input, n_vocab)

    return result

def total_flops(d_embed, d_ffn, ln_bias, attn_bias, mlp_bias, activation,
                   n_head, n_kvhead, n_vocab, n_layer, kvcache, type, prompt_len, output_len):
    # Account for the Prefill phase (this doesn't affect non-caching types)
    total_flops = flops_forward(prompt_len, d_embed, d_ffn, ln_bias, attn_bias, mlp_bias, activation,
                                n_head, n_kvhead, n_vocab, n_layer, False, type)

    # Count the remaining iterations
    for d_input in range(prompt_len + 1, prompt_len + output_len):
        total_flops += flops_forward(d_input, d_embed, d_ffn, ln_bias, attn_bias, mlp_bias, activation,
                                     n_head, n_kvhead, n_vocab, n_layer, kvcache, type)

    return total_flops

class FlopCounter():
    def __init__(self, modelpath="../../models/gpt2"):
        self.modelpath = modelpath
        self.params = None
        self.flops = None

        self.fill_params()

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
        synonyms = [["geglu", "gegelu"],
                    ["swish", "silu"]]

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
                            "kvcache": False, "type": "encoder-decoder"},
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

    def fill_params(self):
        # 1. See if the model is known and if so, return its parameters.
        # 2. If not, then load the model config and parse as many parameters as possible
        # 3. Attempt to infer missing data from parsed fields

        # Step 1
        got_updated, params = self.get_params_for_known_models(self.modelpath)
        if got_updated:
            self.params = params
            return

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
        params["type"] = self.parse_config(config, "is_decoder")

        # Step 3
        self.params = self.infer_params(params)

    def calc_complexity(self, prompt_len, output_len):
        p = self.params
        self.flops = total_flops(p["d_embed"], p["d_ffn"], p["ln_bias"], p["attn_bias"], p["mlp_bias"],
                                 p["activation"], p["n_head"], p["n_kvhead"], p["n_vocab"], p["n_layer"],
                                 p["kvcache"], p["type"], prompt_len, output_len)

    def get_flops(self):
        return self.flops

if __name__ == "__main__":
    print("Phi")
    fc = FlopCounter("../../models/Phi-3-small-8k-instruct")
    fc.calc_complexity(100, 100)
    print(fc.get_flops())
    print("GPT2")
    fc = FlopCounter("../../models/gpt2")
    print("BERT")
    fc = FlopCounter("../../models/bert-base-uncased")
    print("T5")
    fc = FlopCounter("../../models/t5-small")
    print("Teuken")
    fc = FlopCounter("../../models/Teuken-7B-instruct-research-v0.4")
