#!/bin/bash
red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

# global setting
USR="ltan9687"

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

QUOTA_MAX=2
QUOTA_LEFT="not-set"
function get_left_quota() {
    local used=0

    stat=`qstat -u ${USR}`
    if [[ "$stat" != "" ]]; then
        IFS=$'\n' read -rd '' -a stat <<< "${stat}"
        for s in "${stat[@]}"; do
            IFS=" " read -ra s <<< "$s"
            if [[ "${s[1]}" != "${USR}" ]]; then continue; fi

            job_id="${s[0]%.pbs*}"
            job_q="${s[2]}"
            job_stat="${s[9]}"
            # if [[ ${job_q} == "alloc-dt" ]]; then used=$(($used+1)); fi
            if [[ ${job_q} == "alloc-dt" ]]; then ((used++)); fi
        done
    fi

    QUOTA_LEFT=$(( ${QUOTA_MAX} - ${used} ))
    echo -e "used quota = ${used}/${QUOTA_MAX}, ${QUOTA_LEFT} left"
    if (( $used > $QUOTA_MAX )); then echo -e "${red_start} >>>>>>> overusing gpu !!! <<<<<<<<<<< ${red_end}"; fi
}


ID_ARR_gpu=()
ID_ARR_dt=()
function submit_jobs() {
    get_left_quota

    for (( i=0; i<$N; i++)); do
        if (( $QUOTA_LEFT > 0 )); then
            ID_ARR_dt+=(`qsub -q alloc-dt $SCRIPT`)
            (( QUOTA_LEFT-- ))
            if (( $OVERSHOOT == 1 )); then
                ID_ARR_gpu+=(`qsub $SCRIPT`)
            fi
        else
            ID_ARR_gpu+=(`qsub $SCRIPT`)
        fi
    done
}

STAT_ARR=""
function get_submit_stat() {
    local job_n=$(( ${#ID_ARR_gpu[@]} + ${#ID_ARR_dt[@]} ))
    if (( $OVERSHOOT == 1 )); then job_n=$(( $job_n - ${#ID_ARR_dt[@]} )); fi
    if (( $job_n != $N )); then echo -e "${red_start}something wrong?...${red_end}"; fi
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
    while (( $cnt < $N )); do
        get_submit_stat # cnt running jobs
        echo -e "${STAT_ARR[@]}"
        for s in "${STAT_ARR[@]}"; do
            parse_submit_stat $s
            if [[ ${JOB_STAT} == "R" ]]; then cnt=$(($cnt+1)); fi
        done
        sleep 2
    done

    get_submit_stat # del non-runing jobs
    for s in "${STAT_ARR[@]}"; do
        parse_submit_stat $s
        if [[ ${JOB_STAT} == "R" ]]; then ((cnt++)); fi
        if [[ ${JOB_STAT} != "R" || (($cnt > $N)) ]]; then `qdel $JOB_ID`; fi
    done
}


function remove_previous_submit() {
    pattern="*.o[0-9]*"

    local files=`ls -l $pattern 2>/dev/null`
    if [[ "$files" == "" ]]; then echo -e "no previous submits found"; return; fi

    echo -e "found files from previous submits..."
    echo -e "$files"
    read_ynflag "${red_start}cleaning? [y/n]${red_end}"
    if [ "${YN_FLAG}" == "y" ]; then
        rm $pattern
        echo -e "${green_start}done${green_end}"
    else
        echo -e "${green_start}not deleted${green_end}"
    fi
}


function help_content() {
    echo -e "Useage: ./pbs_submit.sh -[sno] [OPTARG]
        -s  the submit_script.sh to submit actual job (will try cleaning the submit dir)
        -n  number of submit to be made (default to 1)
        -o  if overshoot (default no overshoot)
        -c  the submit dir to clean (do NOT specify -c when using -s ... -c will be ignored)"
}

# -o: start listing short args
# --long: start listing long args 
# $@: fetch stdin string into an array
ARGS=`getopt -o s:n:c:o -- "$@"`
eval set -- "${ARGS}" # re-allocate parsed args (key-val) to $1, $2, ...

# solve first-class citizen (the default)
N=1
OVERSHOOT=0 # not overshooting
while true; do
    case ${1} in
        -s)
            SCRIPT=$(readlink -f $2)
            SCRIPT_DIR=`dirname ${SCRIPT}`
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
        -c)
            if [[ "$SCRIPT" == "" ]]; then 
                SCRIPT_DIR=$2
            else 
                echo -e "${red_start}do NOT specify -c when using -s ... -c is ignored${red_end}"
            fi
            shift 2
            ;;
        --)
            break
            ;;
        *)
            shift 1
            ;;
    esac
done

# execute environment
if [[ "$SCRIPT_DIR" == "" ]]; then
    help_content
    exit 1
fi

cd "$SCRIPT_DIR"
remove_previous_submit # with check

if [[ "$SCRIPT" != "" ]]; then
    echo -e "submitting ${green_start}${SCRIPT}${green_end}"
    submit_jobs

    if (( $OVERSHOOT == 1)); then del_overshoot; fi
    echo -e "${green_start}finish${green_end}"
fi

sleep 2
echo -e "`qstat -u ${USR}`"