
# BUG LOG

## 1. Mismatch Between State Dictionary Parameters and the Model Architecture During Checkpoint Loading
When running dist_test.sh or test.py (Using [SRCNN](https://arxiv.org/abs/1501.00092 "Image Super-Resolution Using Deep Convolutional Networks") as an example), the following issues may occur when loading a pre-trained checkpoint:

```shell
"The model and loaded state dict do not match exactly

unexpected key in source state_dict: generator.module.conv1.weight, generator.module.conv1.bias, generator.module.conv2.weight, generator.module.conv2.bias, generator.module.conv3.weight, generator.module.conv3.bias

missing keys in source state_dict: generator.conv1.weight, generator.conv1.bias, generator.conv2.weight, generator.conv2.bias, generator.conv3.weight, generator.conv3.bias"
```

Problem Analysis:
- Initially, you might suspect that the network model and the state_dict saved in the checkpoint do not match, but this is not the case.
- The issue is mainly caused by the parameter revise_keys=[(r'^module//.', '')] in the function _load_checkpoint_to_model (around line 585) in mmengine.runner.checkpoint.py.
- '^module//.' is a regular expression pattern that aims to replace keys like "generator.module.conv1.weight" with "generator.conv1.weight", effectively removing "module." from "generator.module.conv1.weight".
- However, since "generator.module.conv1.weight" does not begin with "module.", it doesn't match the pattern '^module//.'.

Solution:
- In the mmengine.runner.runner.py, within the load_checkpoint function of the Runner class (around line 2111), replace the parameter revise_keys=[(r'^module//.', '')] with revise_keys=[(r'/bmodule.', '')]. This change ensures that keys like "generator.module.conv1.weight" will correctly be replaced with "generator.conv1.weight", effectively removing "module.".

This solution should help resolve the key mismatch when loading checkpoints.