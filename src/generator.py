from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Generator:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(self, prompt, max_new_tokens):
        inputs = self.tokenizer(prompt, return_tensors = "pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = 0.7)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
        return response