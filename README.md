[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) PaddleLSD
===

This is a line segment detection package based on PaddlePaddle.

# Install
```shell
git clone https://github.com/lh9171338/PaddleLSD.git
cd PaddleLSD
python -m pip install -r requirements.txt
python -m pip install -v -e .
```
# Train

```shell
sh train.sh <config>
```

# Test
```shell
sh test.sh <config>
```

# Metric

| model | config | msAP | sAP5 | sAP10 | sAP15 | mAPJ | FPS |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LPN | [config](../configs/wireframe_120ep_shnet_lpn.yaml) | 57.6 | 54.5 | 58.2 | 60.0 | 54.5 | 22 |
| LPN-TwoStage | [config](../configs/wireframe_120ep_shnet_lpn.yaml) | 58.9 | 55.9 | 59.5 | 61.1 | 54.5 | 26 |
| RepLPN | [config](../configs/wireframe_120ep_repshnet_replpn.yaml) | 58.1 | 54.8 | 58.9 | 60.7 | 54.8 | 47 |
| RepLPN-DCN | [config](../configs/wireframe_120ep_repshnet_replpn_dcn.yaml) | 61.3 | 58.5 | 61.9 | 63.6 | 55.9 | 45 |
| RepLPN-DDCN | [config](../configs/wireframe_120ep_repshnet_replpn_ddcn.yaml) | 57.3 | 54.1 | 57.9 | 59.8 | 53.8 | 45 |
