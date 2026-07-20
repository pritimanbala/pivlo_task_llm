# Notes

The best current configuration keeps the original byte tokenizer, model size, context length, batching, Adam optimizer, and checkpoint format unchanged.
The only active modeling experiment is GPT-2 style initialization.
Embeddings and standard linear layers use Normal(mean=0, std=0.02), which is a more conservative starting scale than the original std=0.05 baseline.
The attention output projection and MLP output projection use residual-scaled initialization with std = 0.02 / sqrt(2 * n_layer).
Those two projections are scaled because their outputs are added back into the residual stream.
This should reduce early residual magnitude growth and make optimization more stable during the 2,000-step training budget.
Weight tying, AdamW, LR scheduling, and gradient clipping were not kept because they did not improve development BPB in prior experiments.
The configuration preserves evaluate.py compatibility and remains under the 2,000,000-parameter cap.
