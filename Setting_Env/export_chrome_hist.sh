#!/bin/bash

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

function help_content() {
    echo "\$1 = path-to-history \$2 = output-dir ... besides, will output as \"history.csv\""
}

hist_f=$1
output=$2

if [ $# == 0 ]; then # no arg provided
    help_content
    exit 1
fi

if [ "$hist_f" == "" ]; then # path-to-history not provided
    help_content
    exit 1
fi
read_ynflag "$hist_f is the path to chrome history file? [y/n]" # confirm path-to-histpry
if [ "$YN_FLAG" != "y" ]; then
    echo "abort"
    exit 1
fi

if [ "$output" == "" ]; then # default output = $HOME
    output=${HOME}
fi
read_ynflag "$output is the path to chrome history file? [y/n]" # confirm
if [ "$YN_FLAG" != "y" ]; then
    echo "abort"
    exit 1
fi

# only export the first page of history
sqlite3 ${hist_f} <<EOF
.headers on
.mode csv
.output "$output/history.csv"
SELECT datetime(last_visit_time/1000000-11644473600,'unixepoch','localtime'), url FROM urls ORDER BY last_visit_time DESC
EOF