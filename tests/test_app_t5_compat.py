"""Local app preserves an old untied T5 checkpoint, not just direct HF loading."""
import torch
from safetensors.torch import save_file
from transformers import T5Config, T5ForConditionalGeneration
from looking_glass.app_runtime import LocalModelRuntime
from test_app_runtime import tokenizer


def test_local_app_retains_untied_t5_weights_logits_and_capture(tmp_path):
    path = tmp_path / 'checkpoint'; path.mkdir()
    config = T5Config(vocab_size=16, d_model=8, d_ff=16, d_kv=4, num_layers=1,
                      num_decoder_layers=1, num_heads=2, tie_word_embeddings=False,
                      decoder_start_token_id=0, dropout_rate=0, eos_token_id=None)
    config.tie_word_embeddings = False
    config._attn_implementation = 'eager'
    model = T5ForConditionalGeneration(config).eval()
    model.set_input_embeddings(model.shared)
    config.architectures = ['T5ForConditionalGeneration']
    config.to_json_file(path / 'config.json')
    excluded = {'encoder.embed_tokens.weight', 'decoder.embed_tokens.weight'}
    save_file({k:v.clone() for k,v in model.state_dict().items() if k not in excluded},
              str(path / 'model.safetensors'), metadata={'format':'pt'})
    tokenizer(path)
    runtime = LocalModelRuntime(); runtime.load(str(path), device='cpu')
    assert runtime.model.config.tie_word_embeddings is False
    assert runtime.model.lm_head.weight is not runtime.model.shared.weight
    ids, dec = torch.tensor([[5,6]]), torch.tensor([[0,5]])
    with torch.inference_mode():
        assert torch.equal(model(input_ids=ids, decoder_input_ids=dec).logits,
                           runtime.model(input_ids=ids, decoder_input_ids=dec).logits)
    assert runtime.run('hello world', {'max_new_tokens':1})['coverage']['complete']
