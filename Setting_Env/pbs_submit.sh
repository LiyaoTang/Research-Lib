#!/bin/bash
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

# global setting
USR="ltan9687"

function help_content() {
    echo "Useage: ./pbs_submit.sh -[sno] [OPTARG]
        -s  the submit_script.sh to submit actual job
        -n  number of submit to be made
        -o  if overshoot"
}

QUOTA_MAX=2
QUOTA_LEFT="not-set"
function get_left_quota() {
    local used=0

    stat=`qstat -u ${USR}`
    IFS=$'\n' read -rd '' -a stat <<< "${stat}"

    for s in "${stat[@]}"; do
        IFS=" " read -ra s <<< "$s"
        if [[ "${s[1]}" != "${USR}" ]]; then continue; fi

        job_id="${s[0]%.pbs*}"
        job_q="${s[2]}"
        job_stat="${s[9]}"
        if [[ ${job_q} == "alloc-dt" ]]; then used=$(($used+1)); fi
    done

    QUOTA_LEFT=$(( ${QUOTA_MAX} - ${used} ))
    echo "used quota = ${used}/${QUOTA_MAX}, ${QUOTA_LEFT} left"
    if (( $used > $QUOTA_MAX )); then echo " >>>>>>> overusing gpu !!! <<<<<<<<<<< "; fi
}


ID_ARR_gpu=()
ID_ARR_dt=()
function submit_jobs() {
    get_left_quota
    echo "submitting..."
    
    for (( i=0; i<$N; i++)); do
        if $(( $QUOTA_LEFT > 0 )); then
            ID_ARR_dt+=(`qsub -q alloc-dt $SCRIPT`)
            (( QUOTA_LEFT-- ))
            if $(( $OVERSHOOT == 1 )); then
                ID_ARR_gpu+=(`qsub $SCRIPT`)
            fi
        else
            ID_ARR_gpu+=(`qsub $SCRIPT`)
        fi
    done
}

STAT_ARR=""
function get_submit_stat() {
    if $(( ${#ID_ARR_gpu[@]} != $N )); then echo "${red_start}something wrong?...${red_end}"; fi
    local ids=("${ID_ARR_gpu[@]}" "${ID_ARR_dt[@]}")
    local stat=`qstat "${ids[@]}"`
    IFS=$'\n' read -rd '' -a STAT_ARR <<< "${stat}"
}

JOB_ID=""
JOB_Q=""
JOB_STAT=""
function parse_submit_stat() {
    local s=$1
    IFS=" " read -ra s <<< "$s"
    if [[ "${s[2]}" != "${USR}" ]]; then continue; fi
    JOB_ID="${s[0]%.pbs*}"
    JOB_Q="${s[5]}"
    JOB_STAT="${s[4]}"
}

function del_overshoot() {
    local cnt=0
    while $(( $cnt < $N )); do
        get_submit_stat # cnt running jobs
        for s in "${STAT_ARR[@]}"; do
            parse_submit_stat $s
            if [[ ${JOB_STAT} == "R" ]]; then cnt=$(($cnt+1)); fi
        done
        sleep 2
    done

    get_submit_stat # del non-runing jobs
    for s in "${STAT_ARR[@]}"; do
        parse_submit_stat $s
        if [[ ${JOB_STAT} == "R" ]]; then cnt=$(($cnt+1)); fi
        if [[ ${JOB_STAT} != "R" || (($cnt > $N)) ]]; then `qdel $JOB_ID`; fi
    done
}

# execute environment
if [ $# == 0 ]; then
    help_content
    exit 1
fi

# -o: start listing short args
# --long: start listing long args 
# $@: fetch stdin string into an array
ARGS=`getopt -o s:n: -- "$@"`
eval set -- "${ARGS}" # re-allocate parsed args (key-val) to $1, $2, ...

# solve first-class citizen (the default)
N=1
OVERSHOOT=0 # not overshooting
while true; do
    case ${1} in
        -s)
            SCRIPT=$(realpath $2)
            shift 2
            ;;
        -n)
            N=$2
            shift 2
            ;;
        -o)
            OVERSHOOT=1
            shift 1
            ;;
        --)
            break
            ;;
        *)
            shift 1
            ;;
    esac
done

submit_jobs
del_overshoot
echo "${green_start}finish${green_end}"