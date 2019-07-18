#!/bin/bash
# set up python3 env after changing sys default python from 3.5->3.6

a=$(find /usr/lib/python3/dist-packages -name '*35m*so')
b=$(echo $a | tr 35m 36m)
IFS=' ' read -r -a a <<< $a
IFS=' ' read -r -a b <<< $b

for ((i=0;i<${#a[@]};++i)); do
    ln -s "${a[i]}" "${b[i]}"
done
