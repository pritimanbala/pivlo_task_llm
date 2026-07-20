# Hi interviewer :)
I have chosen the task for training an llm. So here in this file i will be going to share what i have done and how did i thought of approaching it.

TBH, I looked at the problem, I understood it, and then i went to ChatGPT. Before that I downloaded the file in my local, pushed to my repo, and then used codex to summarize the state of the code.

Then i put that in chat GPT and then thought of doing the following change, which i also wanted to do by hypothesis testing. 

| Priority | File                   | Issue                                                | Why it hurts                                                                        | Suggested Fix                                                    | Estimated BPB Gain | Risk        |
|:--------:|:-----------------------|:-----------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------|:------------------:|:------------|
|    1     | `starter/train.py`     | Constant LR, no warmup/decay                         | Wastes part of the 2,000-step budget and may converge poorly                        | Add warmup + cosine or linear decay schedule                     | 0.02–0.08          | Low         |
|    2     | `starter/tokenizer.py` | Raw byte tokenizer                                   | Multi-byte text consumes more tokens; model must learn subword structure from bytes | Train a lossless byte-fallback tokenizer on provided corpus only | 0.05–0.25          | Medium/High |
|    3     | `starter/model.py`     | Possibly underuses 2M parameter cap                  | Model capacity may be too low for best BPB                                          | Tune width/depth/head count under cap and CPU budget             | 0.05–0.15          | Medium      |
|    4     | `starter/model.py`     | Context window only 128 tokens                       | Limits usable left context, especially with byte tokens                             | Increase `block_size` if CPU budget allows                       | 0.02–0.10          | Medium      |
|    5     | `starter/train.py`     | Small default batch size                             | Fewer tokens per optimization step; noisier gradients                               | Increase batch size if CPU/memory allow                          | 0.02–0.08          | Low/Medium  |
|    6     | `starter/model.py`     | Plain fixed-std initialization                       | May be poorly scaled for transformer residual depth                                 | Use GPT-style scaled initialization                              | 0.01–0.06          | Low         |
|    7     | `starter/train.py`     | Adam instead of AdamW, no weight decay               | Weaker regularization/generalization                                                | Use AdamW with reasonable weight decay                           | 0.00–0.04          | Low         |
|    8     | `starter/train.py`     | No gradient clipping                                 | Occasional unstable updates can hurt final checkpoint                               | Add norm clipping before optimizer step                          | 0.00–0.03          | Low         |
|    9     | `starter/model.py`     | Untied token/output embeddings                       | Uses extra params and may generalize worse                                          | Set `tie_weights=True`, especially with larger vocab             | 0.00–0.04          | Low         |
|   10     | `starter/train.py`     | Random sampling only                                 | May repeat data and miss spans within limited steps                                 | Use better packed/epoch-aware batching                           | 0.00–0.04          | Medium      |
|   11     | `starter/train.py`     | No validation/best checkpoint                        | Cannot select best checkpoint or detect overfitting                                 | Add optional dev eval/checkpoint selection                       | 0.00–0.05          | Medium      |
|   12     | `starter/train.py`     | Saves only final checkpoint                          | Final checkpoint may not be best                                                    | Save periodic or best checkpoint                                 | 0.00–0.05          | Low         |
|   13     | `starter/model.py`     | Dropout disabled                                     | May overfit, though may also help fast fitting                                      | Tune small dropout only if validation supports it                | -0.02–0.03         | Medium      |
|   14     | `starter/model.py`     | No explicit `T <= block_size` assertion              | Indirect error if sequence too long                                                 | Add clear assertion                                              | 0.00               | Low         |
|   15     | `starter/model.py`     | Unused `math` import                                 | Cleanliness only                                                                    | Remove import                                                    | 0.00               | Low         |
|   16     | `starter/evaluate.py`  | Overlapping evaluation recomputes context            | Slower evaluation                                                                   | Add exact KV-cache support only if needed                        | 0.00               | Medium/High |
|   17     | `starter/evaluate.py`  | `weights_only=True` may be PyTorch-version-sensitive | Possible compatibility risk                                                         | Confirm grader PyTorch version before changing                   | 0.00               | Unknown     |



ik it looks bad, but in a nutshell, you can understand what i wanted to implement. and the main main things which i could have implemented are there in the file itself.

# 1. Without any updates in the code done by me ->
corpus: 7,318,589 bytes -> 7,318,589 tokens (vocab 256)
model: 1,339,840 params
step     1  loss 5.6104  (239 ms/step)
step   100  loss 3.5168  (201 ms/step)
step   200  loss 2.2816  (199 ms/step)
step   300  loss 2.2310  (203 ms/step)
step   400  loss 2.1956  (205 ms/step)
step   500  loss 2.1879  (204 ms/step)
step   600  loss 2.1326  (203 ms/step)
step   700  loss 2.0924  (204 ms/step)
step   800  loss 2.0783  (204 ms/step)
step   900  loss 2.0802  (205 ms/step)
step  1000  loss 2.0068  (205 ms/step)
step  1100  loss 1.9990  (205 ms/step)
step  1200  loss 1.9715  (206 ms/step)
step  1300  loss 1.9377  (207 ms/step)
step  1400  loss 1.8921  (206 ms/step)
step  1500  loss 1.8692  (205 ms/step)
step  1600  loss 1.8763  (205 ms/step)
step  1700  loss 1.8851  (206 ms/step)
step  1800  loss 1.8355  (206 ms/step)
step  1900  loss 1.8834  (206 ms/step)
step  2000  loss 1.8627  (207 ms/step)
saved ckpt.pt  (414s total)
{"bpb": 2.545, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159226, "tokens_scored": 159225}

# 2. Second Update 
Then i thought, the model was using its basic work like having constant LR, Adam, etc. I tried to apply the following thing: 

1. Replace torch.optim.Adam with torch.optim.AdamW.

2. Add a learning-rate schedule:
   - linear warmup for the first 5% of training steps
   - cosine decay afterwards
   - minimum LR = 10% of initial LR

3. Add gradient clipping using
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

I thought they will help me, but they did worse:

After my first update ->
corpus: 7,318,589 bytes -> 7,318,589 tokens (vocab 256)
model: 1,339,840 params
step     1  loss 5.6104  (347 ms/step)
step   100  loss 2.8646  (247 ms/step)
step   200  loss 2.2551  (253 ms/step)
step   300  loss 2.2303  (243 ms/step)
step   400  loss 2.1996  (232 ms/step)
step   500  loss 2.1940  (224 ms/step)
step   600  loss 2.1425  (224 ms/step)
step   700  loss 2.1028  (221 ms/step)
step   800  loss 2.0901  (217 ms/step)
step   900  loss 2.0897  (217 ms/step)
step  1000  loss 2.0040  (218 ms/step)
step  1100  loss 1.9818  (218 ms/step)
step  1200  loss 1.9366  (219 ms/step)
step  1300  loss 1.8873  (219 ms/step)
step  1400  loss 1.8306  (219 ms/step)
step  1500  loss 1.7922  (219 ms/step)
step  1600  loss 1.7905  (220 ms/step)
step  1700  loss 1.7827  (219 ms/step)
step  1800  loss 1.7288  (218 ms/step)
step  1900  loss 1.7552  (218 ms/step)
step  2000  loss 1.7210  (219 ms/step)
saved ckpt.pt  (439s total)
{"bpb": 2.3787, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159226, "tokens_scored": 159225}


Then I went to claude(as usual :-) and got the following answer which i also feel is correct.

1. Warmup was too long. With only 2,000 steps, a 5% warmup means 100 steps spent at a low learning rate. The baseline's constant LR may simply be more effective in such a short training run.

2. Cosine decay reduced the LR too aggressively. By the last several hundred steps, the learning rate may have been too small to continue making good progress.

3. Weight decay (AdamW) may have been unnecessary for a model trained only 2,000 steps on a relatively small corpus.

4. Gradient clipping probably wasn't the culprit—it usually only matters if gradients explode.

So,
* Hypothesis: AdamW + warmup + cosine decay + gradient clipping will improve convergence.

* Result: Training loss and BPB both became worse.

* Conclusion: Under a strict 2,000-step budget, the baseline optimizer configuration appears better. The additional scheduling and regularization likely reduced effective optimization more than they helped.

So what i have done now is:

1. Restored everything all i changed to base line.
- Restore torch.optim.Adam.
- Remove the learning-rate scheduler.
- Remove gradient clipping.
- Restore the original training behavior.
- I kept the cosine decay with warmup and wanted to see if it helps.

2. Now i am trying to do one new experiment.

* Experiment ->
Enable weight tying between the token embedding and the language modeling head.

then i thought that why the tie_weights = false?

i searchec about it and found that, tie_weights tells the model to try to replicate the embideeding matrix. but if i do it true, it will try to predict the output matrix instead of embidding.

Note -> my parameters will change but it will be close to the initial parameters only


i did that. and here is the result of my approach ->
# 3. Third Update
corpus: 7,318,589 bytes -> 7,318,589 tokens (vocab 256)
model: 1,298,880 params
step     1  loss 5.5647  (242 ms/step)
step   100  loss 3.8501  (171 ms/step)
step   200  loss 2.4274  (176 ms/step)
step   300  loss 2.2780  (185 ms/step)
step   400  loss 2.2155  (189 ms/step)
step   500  loss 2.2041  (202 ms/step)
step   600  loss 2.1494  (213 ms/step)
step   700  loss 2.1099  (218 ms/step)
step   800  loss 2.1003  (222 ms/step)
step   900  loss 2.1119  (226 ms/step)
step  1000  loss 2.0436  (228 ms/step)
step  1100  loss 2.0424  (231 ms/step)
step  1200  loss 2.0208  (233 ms/step)
step  1300  loss 1.9931  (233 ms/step)
step  1400  loss 1.9463  (231 ms/step)
step  1500  loss 1.9257  (229 ms/step)
step  1600  loss 1.9359  (228 ms/step)
step  1700  loss 1.9486  (226 ms/step)
step  1800  loss 1.9022  (224 ms/step)
step  1900  loss 1.9572  (227 ms/step)
step  2000  loss 1.9411  (229 ms/step)
saved ckpt.pt  (457s total)
{"bpb": 2.6576, "n_params": 1298880, "steps": 2000, "tokens_in_eval": 159226, "tokens_scored": 159225}


# 4. Fourth Update
Now i removed the cosine warmup and here is the result without it.

corpus: 7,318,589 bytes -> 7,318,589 tokens (vocab 256)
model: 1,298,880 params
step     1  loss 5.5647  (161 ms/step)
step   100  loss 2.8126  (181 ms/step)
step   200  loss 2.2319  (192 ms/step)
step   300  loss 2.2091  (213 ms/step)
step   400  loss 2.1868  (219 ms/step)
step   500  loss 2.1850  (217 ms/step)
step   600  loss 2.1341  (215 ms/step)
step   700  loss 2.0951  (213 ms/step)
step   800  loss 2.0829  (212 ms/step)
step   900  loss 2.0892  (211 ms/step)
step  1000  loss 2.0118  (209 ms/step)
step  1100  loss 1.9981  (211 ms/step)
step  1200  loss 1.9589  (215 ms/step)
step  1300  loss 1.9046  (214 ms/step)
step  1400  loss 1.8449  (214 ms/step)
step  1500  loss 1.8076  (212 ms/step)
step  1600  loss 1.8058  (214 ms/step)
step  1700  loss 1.8012  (217 ms/step)
step  1800  loss 1.7470  (217 ms/step)
step  1900  loss 1.7734  (218 ms/step)
step  2000  loss 1.7386  (217 ms/step)
saved ckpt.pt  (435s total)
{"bpb": 2.3941, "n_params": 1298880, "steps": 2000, "tokens_in_eval": 159226, "tokens_scored": 159225}


So, 
Parameter count decreased by 40,960, exactly what we'd expect from tying the embedding and output head.
Training loss became slightly worse (1.7210 → 1.7386).
BPB became slightly worse (2.3787 → 2.3941).

Now we can infer that,
1. Weight tying is not helping this particular model. For large models, we can do that, but for this small model, i dont feel it can bring that good affect. 
2. The model actually benefits slightly from having separate embeddings and output projections.
3. Now i am pakka thinking that the bottleneck is probably model capacity or context.


Till now, 
Hypothesis: Weight tying would improve parameter efficiency and generalization.

Observation: Parameters reduced by ~41k, but BPB increased from 2.3787 to 2.3941.

Conclusion: For this small byte-level model, the extra flexibility of untied embeddings outweighed the parameter savings. Weight tying was therefore reverted.


Now i will try the following things -> 
1. Better initialization
2. Increase context length, block size = 128 -> 192





