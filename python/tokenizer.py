from transformers import AutoTokenizer

def get_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return tokenizer