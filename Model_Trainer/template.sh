#!/bin/bash

if [ "$BASH_VERSION" = '' ]; then
    echo "warining: should run by bash"
    exit
fi

trap "exit" INT TERM # convert other temination signal to EXIT
trap "kill 0" EXIT # crtl-C stop the current & background script

