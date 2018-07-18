#!/bin/bash

ngpu=$(nvidia-smi topo -m | grep '^GPU' | wc -l)
prefix="/tmp/gpu_lock_"

sleep $[ ( $RANDOM % 5 ) ].$[ ( $RANDOM % 1000 ) + 1 ]s

while :; do
    for ((i=1;i<=${ngpu};i++)); do
        echo "testing gpu $i of $ngpu"
        if [ ! -f ${prefix}${i} ]; then
            touch ${prefix}${i} 
            echo "working on gpu $i"
            echo $@
            export CUDA_VISIBLE_DEVICES=$(expr ${i} - 1)
            "$@"
            ret=$?
            echo "work done"
            rm ${prefix}${i}
            exit $ret
        fi
    done
    echo "all gpu occupied"
    sleep 10
done


