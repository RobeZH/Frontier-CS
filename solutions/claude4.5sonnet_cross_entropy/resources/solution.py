class Solution:
    def solve(self, spec_path: str = None) -> dict:
        code = """
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
    ],
    key=['N'],
)
@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    output_ptr,
    M,
    N,
    logits_stride_0,
    logits_stride_1,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    target = tl.load(targets_ptr + row_idx)
    row_start = logits_ptr + row_idx * logits_stride_0
    
    # First pass: find maximum for numerical stability
    max_val = float('-inf')
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        logits = tl.load(row_start + cols * logits_stride_1, mask=mask, other=float('-inf'))
        block_max = tl.max(logits)
        max_val = tl.maximum(max_val, block_max)
    
    # Second pass: compute log-sum-exp and extract target logit
    sum_exp = 0.0
    target_logit = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        logits = tl.load(row_start + cols * logits_stride_1, mask=mask, other=0.0)
        
        # Extract target logit
        target_mask = cols == target
        target_logit += tl.sum(tl.where(target_mask & mask, logits, 0.0))
        
        # Accumulate exp sum
        sum_exp += tl.sum(tl.where(mask, tl.exp(logits - max_val), 0.0))
    
    # Compute cross entropy loss
    log_sum_exp = tl.log(sum_exp) + max_val
    loss = -target_logit + log_sum_exp
    
    tl.store(output_ptr + row_idx, loss)

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    M, N = logits.shape
    output = torch.empty(M, dtype=torch.float32, device=logits.device)
    
    grid = (M,)
    cross_entropy_kernel[grid](
        logits,
        targets,
        output,
        M,
        N,
        logits.stride(0),
        logits.stride(1),
    )
    
    return output
"""
        return {"code": code}
