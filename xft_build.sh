#!/bin/bash
bash ./lite/tools/build_linux.sh \
    --arch=x86 \
    --with_baidu_xpu=ON \
    --kunlunxin_xpu_sdk_env=ubuntu_x86_64 \
    --kunlunxin_xpu_sdk_url=https://baidu-kunlun-product.su.bcebos.com/KL-SDK/klsdk-dev_paddle \
    --kunlunxin_xpu_xft_root=/liwei127/codes/baidu/xpu/xft/build/xft_install/
