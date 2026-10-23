import pytest
torch = pytest.importorskip("torch")
from agsgpt.model import AGsGPT, CharTokenizer, LoRALinear, adapter_state, apply_lora

def _model():
    torch.manual_seed(0)
    return AGsGPT(20, d=32, h=4, n_layers=2, ctx=16)

def test_identity_at_init():
    m = _model().eval(); x = torch.randint(0, 20, (2, 10))
    before = m(x)[0]
    apply_lora(m, 4, 4); m.eval()
    assert torch.allclose(before, m(x)[0], atol=1e-6)

def test_only_A_B_trainable():
    m = _model(); n = apply_lora(m, 4, 4)
    names = [k for k, p in m.named_parameters() if p.requires_grad]
    assert names and all(k.rsplit(".", 1)[-1] in ("A", "B") for k in names)
    assert n == sum(p.numel() for _, p in m.named_parameters() if p.requires_grad)

def test_gradients_flow_only_to_adapters():
    m = _model(); apply_lora(m, 4, 4)
    x = torch.randint(0, 20, (1, 12)); _, loss = m(x, x); loss.backward()
    for k, p in m.named_parameters():
        if p.requires_grad: assert p.grad is not None
        else: assert p.grad is None

def test_adapter_state_small_and_complete():
    m = _model(); apply_lora(m, 4, 4); st = adapter_state(m)
    assert len(st) == 2 * 4 * 2 and all(k.endswith((".A", ".B")) for k in st)

def test_tokenizer_roundtrip(tmp_path):
    t = CharTokenizer().fit("hola mundo"); p = tmp_path / "t.json"; t.save(p)
    t2 = CharTokenizer().load(p); assert t2.decode(t2.encode("hola")) == "hola"

def test_lora_scaling():
    lin = torch.nn.Linear(3, 2); l = LoRALinear(lin, rank=2, alpha=4); assert l.scaling == 2.0
