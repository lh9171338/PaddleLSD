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

| model | config | msAP | sAP5 | sAP10 | sAP15 | mAPJ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LPN | [config](../configs/wireframe_120ep_shnet_lpn.yaml) | 58.5 | 55.6 | 59.1 | 60.8 | 54.0 |
