#!/bin/bash

ngpu=4
prefix="lock_"

sleep $[ ( $RANDOM % 3 ) ].$[ ( $RANDOM % 1000 ) + 1 ]s

while :; do
    for ((i=1;i<=${ngpu};i++)); do
        echo "testing gpu $i"
        if [ ! -f ${prefix}${i} ]; then
            echo "working on gpu $i"
            touch ${prefix}${i} 
            echo $@
            export CUDA_VISIBLE_DEVICES=$(expr ${i} - 1)
            "$@"
            ret=$?
            rm ${prefix}${i}
            exit $ret
        fi
    done
    echo "all gpu occupied"
    sleep 10
done


