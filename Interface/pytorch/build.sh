#!/bin/bash

cd ./build
cmake ../ -DCMAKE_PREFIX_PATH=/usr/local/include/libtorch ..
make