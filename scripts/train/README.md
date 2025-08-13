# About Training Scripts

## We follow the basic training scripts from LLaVA-NeXT. It's based on previous LLaVA's training scripts and researchers familiar with LLaVA will find it easy to use.

1. We provide the training scripts for our stage-1 training: ==Reasoning and learning to question==.
> The main change from LLaVA-NeXT is we implemented a data processing function called ==preprocess_qwen_sq==, which is used to unmask question tokens during training.
> To be notice, we are setting `--version qwen_sq` to make sure we are using self-questioning data process.

- `finetune_ov_sq.sh`: This could be seen as the first-stage training script.


2. We provide the training scripts for our stage-2 training: ==Structured preference optimization==.
> The main change here is we implemented a new trainer called ==SPO_trainer==, you may find it in `trl/trainer/spo_trainer.py`
> The train_spo.py implements a new data process to support preferene data.

- `spo_7b.sh`: This could be seen as the second-stage training script.