# Static security and source review

Reviewed the archive contents without executing the Python scripts, opening the HTML report in a browser, installing dependencies, or loading a model checkpoint.

## Security observations

- No obvious shell execution, dynamic `eval`/`exec`, credential access, network requests, or data exfiltration was found in `train.py` or `predict.py`.
- `predict.py` uses `torch.load(checkpoint, ...)`. Treat checkpoint files as executable/untrusted input because pickle-based loading can execute code. The archive contained no checkpoint.
- The exported HTML report references `require.js`, jQuery, and MathJax from external CDNs; it was left out of this repository package.

This is a static review of the supplied files, not a guarantee that third-party dependencies or future checkpoints are safe. Keep endpoint protection enabled and obtain dependencies from trusted sources.

## Functional issues to address before use

- `train.py` hard-codes `epochs = 5`, so the `--epochs` argument is ignored.
- `validation()` hard-codes CUDA for the model and tensors, which fails on CPU-only systems and ignores the selected device.
- `save_checkpoint()` hard-codes the architecture to `vgg16`, even if AlexNet was selected.
- `save_checkpoint()` stores `optimizer.state_dict` as a method reference rather than calling `optimizer.state_dict()`.
- `predict.py` parses `--gpu` as `bool`; command-line text values such as `False` are still truthy in Python.
- The archive lacks the flower dataset, `cat_to_name.json`, and checkpoint required to use the scripts.

No fixes were applied to the original source files in this preparation pass.
