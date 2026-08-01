Loading base model: Qwen/Qwen2.5-1.5B-Instruct
Loading weights: 100% 338/338 [00:01<00:00, 252.88it/s]
trainable params: 18,464,768 || all params: 1,562,179,072 || trainable%: 1.1820
Loading datasets...
Generating train split: 116 examples [00:00, 42466.55 examples/s]
Generating train split: 25 examples [00:00, 12390.12 examples/s]
  train: 116 examples
  val:   25 examples
[transformers] warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
Applying formatting function to train dataset: 100% 116/116 [00:00<00:00, 4248.36 examples/s]
Adding EOS to train dataset: 100% 116/116 [00:00<00:00, 14691.08 examples/s]
Tokenizing train dataset: 100% 116/116 [00:00<00:00, 1420.83 examples/s]
Building labels for train dataset: 100% 116/116 [00:00<00:00, 4787.22 examples/s]
Truncating train dataset: 100% 116/116 [00:00<00:00, 4343.21 examples/s]
Dropping fully masked examples from train dataset: 100% 116/116 [00:00<00:00, 7356.20 examples/s]
Applying formatting function to eval dataset: 100% 25/25 [00:00<00:00, 3923.72 examples/s]
Adding EOS to eval dataset: 100% 25/25 [00:00<00:00, 6545.83 examples/s]
Tokenizing eval dataset: 100% 25/25 [00:00<00:00, 1356.34 examples/s]
Building labels for eval dataset: 100% 25/25 [00:00<00:00, 3078.43 examples/s]
Truncating eval dataset: 100% 25/25 [00:00<00:00, 2943.04 examples/s]
Dropping fully masked examples from eval dataset: 100% 25/25 [00:00<00:00, 4711.64 examples/s]
Starting training...
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': None, 'pad_token_id': 151643}.
{'loss': '3.245', 'grad_norm': '1.883', 'learning_rate': '0.0001917', 'entropy': '2.411', 'num_tokens': '1.171e+04', 'mean_token_accuracy': '0.4659', 'epoch': '0.6897'}
 33% 8/24 [01:40<02:44, 10.31s/it]
  0% 0/4 [00:00<?, ?it/s]
 50% 2/4 [00:02<00:02,  1.02s/it]
 75% 3/4 [00:04<00:01,  1.49s/it]
                                  
{'eval_loss': '1.534', 'eval_runtime': '6.584', 'eval_samples_per_second': '3.797', 'eval_steps_per_second': '0.608', 'eval_entropy': '1.735', 'eval_num_tokens': '1.688e+04', 'eval_mean_token_accuracy': '0.712', 'epoch': '1'}
 33% 8/24 [01:46<02:44, 10.31s/it]
100% 4/4 [00:04<00:00,  1.08s/it]
{'loss': '1.733', 'grad_norm': '1.492', 'learning_rate': '0.000146', 'entropy': '1.915', 'num_tokens': '2.155e+04', 'mean_token_accuracy': '0.6785', 'epoch': '1.276'}
{'loss': '1.253', 'grad_norm': '0.543', 'learning_rate': '7.965e-05', 'entropy': '1.303', 'num_tokens': '3.32e+04', 'mean_token_accuracy': '0.7569', 'epoch': '1.966'}
 67% 16/24 [03:27<01:25, 10.64s/it]
  0% 0/4 [00:00<?, ?it/s]
 50% 2/4 [00:02<00:02,  1.02s/it]
 75% 3/4 [00:04<00:01,  1.50s/it]
                                   
{'eval_loss': '1.244', 'eval_runtime': '6.602', 'eval_samples_per_second': '3.787', 'eval_steps_per_second': '0.606', 'eval_entropy': '1.282', 'eval_num_tokens': '3.376e+04', 'eval_mean_token_accuracy': '0.7632', 'epoch': '2'}
 67% 16/24 [03:33<01:25, 10.64s/it]
100% 4/4 [00:04<00:00,  1.08s/it]
{'loss': '1.171', 'grad_norm': '0.5195', 'learning_rate': '2.243e-05', 'entropy': '1.202', 'num_tokens': '4.305e+04', 'mean_token_accuracy': '0.7681', 'epoch': '2.552'}
100% 24/24 [05:14<00:00, 10.67s/it]
  0% 0/4 [00:00<?, ?it/s]
 50% 2/4 [00:02<00:02,  1.02s/it]
 75% 3/4 [00:04<00:01,  1.50s/it]
                                   
{'eval_loss': '1.228', 'eval_runtime': '6.599', 'eval_samples_per_second': '3.788', 'eval_steps_per_second': '0.606', 'eval_entropy': '1.257', 'eval_num_tokens': '5.063e+04', 'eval_mean_token_accuracy': '0.7653', 'epoch': '3'}
100% 24/24 [05:20<00:00, 10.67s/it]
100% 4/4 [00:04<00:00,  1.08s/it]
{'train_runtime': '321.8', 'train_samples_per_second': '1.082', 'train_steps_per_second': '0.075', 'train_loss': '1.742', 'epoch': '3'}
100% 24/24 [05:21<00:00, 13.41s/it]
Saving adapter to results/lora_adapter
Found 7 files to upload
  Preparing   ████████████████████  7 / 7 ✓
  Uploading   █████████████░░░░░░░  2 / 3 files
  Committing  ░░░░░░░░░░░░░░░░░░░░  0 / 7
  Preparing   ████████████████████  7 / 7 ✓
  Uploading   ████████████████████  3 / 3 files ✓
  Committing  ████████████████████  7 / 7 ✓
No files have been modified since last commit. Skipping to prevent empty commit.
Pushing adapter to Hugging Face Hub: Mehboobali512/trustworthy-slm-agent-qwen2.5-1.5b
Found 7 files to upload
  Preparing   ████████████████████  7 / 7 ✓
  Uploading   █████████████░░░░░░░  2 / 3 files
  Committing  ░░░░░░░░░░░░░░░░░░░░  0 / 7
  Preparing   ████████████████████  7 / 7 ✓
  Uploading   ████████████████████  3 / 3 files ✓
  Committing  ████████████████████  7 / 7 ✓
No files have been modified since last commit. Skipping to prevent empty commit.
Done.
