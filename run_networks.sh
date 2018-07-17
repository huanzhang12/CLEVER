#!/bin/bash

# to run: xargs -I{} -P 4 bash -c ./gpu_check.sh {}

rm lock_*

model="$@"

declare -A batchsize=( ["resnet_v2_50"]="128" ["resnet_v2_101"]="128" ["resnet_v2_152"]="64" ["inception_v1"]="256" ["inception_v2"]="256" ["inception_v3"]="128" ["inception_v4"]="96" ["inception_resnet_v2"]="96" ["vgg_16"]="192" ["vgg_19"]="192" ["mobilenet_v1_025"]="512" ["mobilenet_v1_050"]="512" ["mobilenet_v1_100"]="384" ["nasnet_large"]="26" ["densenet121_k32"]="128" ["densenet169_k32"]="96" ["densenet161_k48"]="64" )

for m in $model; do
    if [ ! ${batchsize[$m]+abc} ]; then
        >&2 echo "model $m is unknown"
        exit 1
    fi
done

for m in $model; do
    for f in 0 25 50 75; do
        b="${batchsize[$m]}"
        output_dir="results/${m}/${m}_${f}"
        echo python3 collect_gradients.py -d imagenet -m ${m} -f ${f} --nthreads 6 -n 25 -s ${output_dir} --batch_size $b
    done
done

