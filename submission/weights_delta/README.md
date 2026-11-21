# weights_delta (not included)

The submitted zip carries `weights_delta/model_delta.safetensors` (1.35 GB): the 118 tensors (language model + lm_head) in
which team taktak's draft-tuned GravityOCR-KO weights (from our earlier submission 278) differ from the stock weights.
`serve.py` (`_merge_weights_delta`) assembles the full model folder from the stock weights in the competition image plus
these tensors before the server starts (untimed). The file is too large for a git repository and is not published here.

Without it, `serve.py` skips the merge and serves the stock weights of the image: same code path, about 3% slower in our
measurements (2.29 s vs 2.35 s for the winner's v17 with and without these weights).
