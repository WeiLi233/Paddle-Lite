#!/bin/bash
 
#export http_proxy=http://agent.baidu.com:8118
#export https_proxy=http://agent.baidu.com:8118
# export LD_LIBRARY_PATH=/home/liwei127/my_data/tools/xpu_toolchain/XTCL/lib/:/home/liwei127/my_data/tools/xpu_toolchain/XTCL/shlib/:${LD_LIBRARY_PATH} 
#export CC=/opt/compiler/gcc-8.2/bin/gcc
#export CXX=/opt/compiler/gcc-8.2/bin/g++
LITE_BUILD_THREADS=$(nproc) bash ./lite/tools/build_linux.sh \
    --with_python=OFF \
    --python_version=3.7 \
    --with_avx=ON \
    --arch=x86 \
    --with_extra=ON 
    ## --with_profile=ON \
    ## --with_precision_profile=ON
    #--baidu_xpu_sdk_env=centos7_x86_64 
    ## --baidu_xpu_sdk_root="${HOME}/my_data/zhupengyang/XPU_SDK/output"
    ## --baidu_xpu_sdk_root=/home/liwei127/my_data/tools/xpu_toolchain/
