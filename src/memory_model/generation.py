import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """Return shape [batch, 1]. Greedy decoding is selected by do_sample=False."""
    if not do_sample or temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and top_p < 1.0:
        if top_p <= 0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        # Keep the first token that crosses p, as well as every token before it.
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)
