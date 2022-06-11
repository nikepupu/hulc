# Modifications

## Code adjustment & Bugs
### 1 skip_frames: 24
### 2. Improved validation loss logging
### 3. Check validation dataset setting

In ArnoldDataModule, there are four datasets: `vision_dataset` and `lang_dataset` for both training and validation

For `vision_dataset`

```
len(train_dataset) == 469
len(val_dataset) == 469
```

For `lang_dataset`

```
len(train_dataset) == 469
len(val_dataset) == 469
```
#### **Fixes**

split into the data with `split.py` and get training and validation set

* Force ArnoldBaseDataset to inherit ABC

In ArnoldDataModule, the number of batches
```
len(train_dataloaders['vis']) == 34
len(train_dataloaders['lang']) == 34
len(combined_val_loaders) == 3
```

### 4. DataLoader didn't set shuffle=True, default is False

# Training Log

## Exp1: 2020-06-10/17-17-43

using default config, 
```yaml
img_lang_matching_clip:false

model:
  - perceptual_encoder: hulc.models.perceptual_encoders.arnold_concat_encoders.ArnoldConcatEncoders
  - plan_proposal: hulc.models.plan_encoders.plan_proposal_net.PlanProposalNetwork
  - plan_recognition: hulc.models.plan_encoders.plan_recognition_net.PlanRecognitionTransformersNetwork
  - distribution:  discrete
  - visual_goal: hulc.models.encoders.goal_encoders.VisualGoalEncoder
  - language_encoder: none
  - language_goal: hulc.models.encoders.goal_encoders.LanguageGoalEncoder
  - action_decoder: hulc.models.decoders.logistic_decoder_rnn.LogisticDecoderRNN
  - optimizer: adam
  - lr_scheduler: constant
  - lang_decoder: none
  - lang_discriminator: none
  - clip_proj: default
```

Test run, best performance
```yaml
[2022-06-10 17:31:25,442][hulc.models.arnold_hulc][INFO] - Start validation epoch 2
[2022-06-10 17:31:40,532][hulc.models.arnold_hulc][INFO] - Validation action_loss_pp: 7.5390580495198565 
[2022-06-10 17:31:40,532][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.2310388187567393 
[2022-06-10 17:31:40,540][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.22151197989781699 
```

## Exp2: 2020-06-10/17-17-43

Adjusted num_workers, batch_size
 best performance
```yaml
[2022-06-10 18:08:19,544][hulc.models.arnold_hulc][INFO] - Start validation epoch 34
[2022-06-10 18:08:25,767][hulc.models.arnold_hulc][INFO] - Validation action_loss_pp: 1.4742100834846497 
[2022-06-10 18:08:25,767][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.10105859239896138 
[2022-06-10 18:08:25,767][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.09998507301012675 
```

loss becomes inf after some epcoh 35.

## Exp3 2020-06-10/18-37-03

Adjusted max_gradient_norm=0.5, fp16=false
 best performance
```yaml
[2022-06-10 19:09:41,600][hulc.models.arnold_hulc][INFO] - Start validation epoch 36
[2022-06-10 19:09:47,588][hulc.models.arnold_hulc][INFO] - Validation action_loss_pp: 3.216231187184652 
[2022-06-10 19:09:47,588][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.10971302787462871 
[2022-06-10 19:09:47,588][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.11710686981678009 
```

## Exp4 2020-06-10/20-07-30

AdamW has bug which causes segmentation fault.

```yaml
lr: 2e-4
max_gradient_norm: 2.0
#Adam with weight_decay=0.01 (L2 norm)
optimizer: adam
lr_scheduler: linear_schedule_with_warmup
```

 best performance
```yaml
[2022-06-10 21:01:16,277][hulc.models.arnold_hulc][INFO] - Start validation epoch 61
[2022-06-10 21:01:22,412][hulc.models.arnold_hulc][INFO] - Validation action_loss_pp: 6.64550797144572 
[2022-06-10 21:01:22,413][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.26163390278816223 
[2022-06-10 21:01:22,413][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.26137475669384 
```

## Exp5 2020-06-10/

Switching to CLIP model, freeze features. It does not converge easily.


 best performance
```yaml
[2022-06-10 23:12:26,031][hulc.models.arnold_hulc][INFO] - Start validation epoch 18
[2022-06-10 23:12:52,149][hulc.models.arnold_hulc][INFO] - Validation action_loss_pp: 4.83213484287262 
[2022-06-10 23:12:52,149][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.1100696638226509 
[2022-06-10 23:12:52,150][hulc.models.arnold_hulc][INFO] - Validation pp_mae_mean: 0.10546934604644775 
```
