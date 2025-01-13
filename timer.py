import time

import torch


class InferenceTimer:
    def __init__(self):
        self.num_output_token = None
        self.start_time = None
        self.end_time = None
        self.setup_time = None
        self.tokenize_time = None
        self.time_to_first_token = None

    def start_timer(self):
        self.start_time = time.perf_counter()

    def end_timer(self):
        self.end_time = time.perf_counter()

    def stop_setup_time(self):
        self.setup_time = time.perf_counter()

    def stop_tokenize_timer(self):
        self.tokenize_time = time.perf_counter()

    def stop_pure_tokenization_timer(self):
        self.pure_tokenization = time.perf_counter()

    def generation_time(self):
        if not self.tokenize_time:
            return self.end_time - self.setup_time
        else:
            return self.end_time - self.tokenize_time

    def token_per_sec(self, output_token):
        if not torch.is_tensor(output_token):
            output_token = torch.FloatTensor(output_token)

        total_sequences, total_token = output_token.shape
        return (total_sequences * total_token) / (self.end_time - self.setup_time)

    def seq_per_sec(self, output_token):
        if not torch.is_tensor(output_token):
            output_token = torch.FloatTensor(output_token)
        total_sequences = output_token.shape[0]
        return total_sequences / (self.end_time - self.setup_time)

    def total_prediction_time(self):
        return self.end_time - self.start_time

    def time_per_token(self, output_token):
        if not torch.is_tensor(output_token):
            output_token = torch.FloatTensor(output_token)

        total_sequences, total_token = output_token.shape
        self.num_output_token = total_sequences * total_token

        return (self.end_time - self.setup_time) / self.num_output_token

    def time_for_setup(self):
        # including tokenize_time
        return self.setup_time - self.start_time

    def time_for_pre_tokenization(self):
        return self.tokenize_time - self.setup_time

    def token_transfer_time(self):
        return self.tokenize_time - self.pure_tokenization
