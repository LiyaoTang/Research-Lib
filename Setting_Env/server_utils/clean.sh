#!/bin/bash
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

YN_FLAG=""
function read_ynflag() {
    local msg=$1
    YN_FLAG=""
    echo -ne "$msg "
    read YN_FLAG
    while [[ $YN_FLAG != 'y' && $YN_FLAG != 'n' ]]; do
        echo -ne "${msg} "
        read YN_FLAG
    done
}

pattern="`dirname $0`/*.o[0-9]*"

echo -e "cleaning ..."
ls -l $pattern

read_ynflag "${red_start}correct? [y/n]${red_end}"
if [ "${YN_FLAG}" == "y" ]; then
    rm $pattern
    echo -e "${green_start}done${green_end}"
else
    echo -e "${green_start}abort${green_end}"
fi