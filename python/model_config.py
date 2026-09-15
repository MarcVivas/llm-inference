import json

class ModelConfig:
    def __init__(self, config) -> None:
        self.hidden_size = getattr(config, "hidden_size", 2560)
        self.num_layers = getattr(config, "num_hidden_layers", 32)
        self.num_heads = getattr(config, "num_attention_heads", 20)
        self.num_kv_heads = getattr(config, "num_key_value_heads", 4)
        self.intermediate_size = getattr(config, "intermediate_size", 6912)
        self.vocab_size = getattr(config, "vocab_size", 152064)
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.max_seq_len = getattr(config, "max_position_embeddings", 2048)
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.rope_theta = getattr(config, "rope_theta", 1000000.0)

    def print_architecture(self):
        print("Architecture Hyperparameters:")
        print(f"  Layers: {self.num_layers}, Hidden Size: {self.hidden_size}, Vocab: {self.vocab_size}")
        print(f"  Heads (Q/KV): {self.num_heads}/{self.num_kv_heads}, Head Dim: {self.head_dim}")
        print(f"  Intermediate Size (SwiGLU): {self.intermediate_size}")
        print(f"  Rope Theta: {self.rope_theta}, RMSNorm Epsilon: {self.rms_norm_eps}")


    def export_json(self, out_file):
        config_dict = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
        }
        config_out = out_file.with_name("model_config.json")
        with open(config_out, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Exported configuration to: {config_out}")
           