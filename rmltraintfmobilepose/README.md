# Raven Training Plugins: Mobile Pose

Use this plugin to learn end-to-end pose estimation.

![screenshot](https://user-images.githubusercontent.com/6625384/131351345-34a6ff81-0441-4743-8058-46a92cfd9822.gif)

## Config

Here's an example config (`train_config.json`) that should be passed to all commands when using this plugin.

```javascript
{
  "dataset": "cyg_nm_norm_2k",
  "metadata": {
    "created_by": "John Smith",
    "comments": "Training model."
  },
  "plugin": {
    "keypoints": 20,
    "batch_size": 64,
    "cache_train_data": false,
    "prefetch_num_batches": 10000,
    "crop_size": 224,
    "shuffle_buffer_size": 1,
    "model_arch": "mobilenet",
    "model_init_weights": "imagenet",
    "model_unet_params": {},
    "phases": [
      {"optimizer":  "Adam", "optimizer_args":  {"learning_rate": 0.001}, "epochs": 1000, "start_layer": "input_1"}
    ],
    "pnp_focal_length": 1422,
    "dropout": 0.0
  }
}
```

## Commands

### Train Pose Model

Train a mobilenet model. Most of the params are adjusted through the `train_config.json`. Comet API is optional.

`ravenml train --config train_config.json tf-mobilepose train --comet <api-key>`

> `ravenml train --config train_config.json tf-mobilepose train  --comet GHJHIHAMLNHNKSVASD`

### Eval Pose Model

Evaluate a trained pose model. Use `--render` to generate visualizations.

`ravenml train --config train_config.json tf-mobilepose eval <model>.h5 <path/to/dataset> -f <focal-length>`

See `scripts/eval.py` for more options.

> `ravenml train --config train_config.json tf-mobilepose eval sota.h5 F:\ravenml\datasets\cygnus_20k_re_norm_mix_drb\test -f 1422`

### Train CullNet (error detection)

See `scripts/cull.py` for more options.

Train a `mobilenetv2_imagenet_mse` cullnet model using `numpy_cut`-masks. Use `cull.npy` to cache error data. `pose_model.h5` is used for generating error data.
> `ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse`

### Eval CullNet

See `scripts/cull.py` for more options.

Eval model `mobilenetv2_imagenet_mse-1609270326-best.h5` on `~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test` data.
> `ravenml train --config train_config.json tf-mobilepose cull pose_model.h5 ~/ravenml/datasets/cygnus_20k_re_norm_mix_drb/test -f 1422 -k numpy_cut -c cull.npy -m mobilenetv2_imagenet_mse -t mobilenetv2_imagenet_mse-1609270326-best.h5`

### Export Model

See `scripts/export_model.py` for more options.

> `ravenml train --config train_config.json tf-mobilepose export_model <model-h5-fn> <tf-SavedModel-fn>`

## Reference

* [MobilePose: Real-Time Pose Estimation for Unseen Objects with Weak Shape Supervision](https://arxiv.org/abs/2003.03522)
* [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
