import torch

B, V, K = 1024, 50000, 2048 # "standard" topk inference workload
num_warmups = 10
repeat_times = 50
dtype = torch.float32 # is this right?

# assume that logits are already on GPU. fair assumption here but were this not the case, the time to move logits from CPU to GPU dominates.
x = torch.randn(B, V, dtype=dtype).cuda()  # output after last linear layer

# warmup
for _ in range(num_warmups):
    torch.topk(x, K, dim=-1)
torch.cuda.synchronize()

start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(repeat_times):
    torch.topk(x, K, dim=-1)
end_event.record()
torch.cuda.synchronize()
print(f"Avg time taken: {start_event.elapsed_time(end_event) / repeat_times:.3f} ms") # ~0.86 ms on PT2.8 H100