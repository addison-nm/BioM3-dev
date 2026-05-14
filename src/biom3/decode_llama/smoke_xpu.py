"""Standalone XPU smoke test for ZpDecoder. Bypasses Lightning entirely."""
import sys, time, argparse, torch

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    ipex = None

from biom3.decode.model import ZpDecoder

def main():
    p = argparse.Namespace(
        lm_path="./weights/LLMs/biogpt-base",
        z_dim=512, prefix_length=10, hidden_dim=2048, dropout=0.1,
    )
    print(f"torch.xpu.is_available: {torch.xpu.is_available()}")
    print(f"torch_xla loaded: {'torch_xla' in sys.modules}")
    print(f"ipex: {ipex.__version__ if ipex else 'MISSING'}")

    model = ZpDecoder(p).to("xpu")
    model.train()
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)
    if ipex:
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16, inplace=True)
        
    B, T = 32, 512
    z_p = torch.randn(B, 512, device="xpu")
    input_ids = torch.randint(0, 40000, (B, T), device="xpu")
    attn = torch.ones(B, T, dtype=torch.long, device="xpu")
    labels = input_ids.clone()

    # warmup
    with torch.amp.autocast(device_type="xpu", dtype=torch.bfloat16):
        out = model(z_p, input_ids, attn, labels)
    out.loss.backward()
    torch.xpu.synchronize()

    # time 5 steps
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad()
        with torch.amp.autocast(device_type="xpu", dtype=torch.bfloat16):
            out = model(z_p, input_ids, attn, labels)
        out.loss.backward()
        opt.step()
    torch.xpu.synchronize()
    dt = (time.time() - t0) / 5
    print(f"avg step: {dt:.2f}s  | xpu mem: {torch.xpu.max_memory_allocated()/1e9:.2f} GB")

if __name__ == "__main__":
    main()
