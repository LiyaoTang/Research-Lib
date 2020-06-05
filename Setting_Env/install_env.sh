#!/bin/bash

red_start="\033[31m"
red_end="\033[0m"
green_start="\033[32m"
green_end="\033[0m"
set -e # exit as soon as any error occur

function help_content() {
    echo "Useage: ./install_env.sh -pi [OPTARG]
        -p  --path      the path to extracted folder for packages of install source
        -i  --install   followed by items to be installed"
}

function exit_script() {
    local err_msg=$1
    echo -e "${red_start}${err_msg}${red_end}"
    exit 1
}

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

RST=""
function read_with_default() {
    local msg=$1
    local default=$2
    local tmp
    RST=$default
    echo -ne "${msg} (default to ${default}) "
    read tmp
    if [ $tmp != "" ]; then
        RST=$tmp
    fi
}

SELECTED="" # global var as return value
function select_items() {
    local items_arr=$1
    local default=0
    local i
    
    if [[ ${#items_arr[@]} -le 0 ]]; then
        exit_script "no available items to be selected, abort"
    fi

    echo -e "select from:"
    for (( i=0; i<${#items_arr[@]}; ++i )); do
        echo -e "${i}.\t${items_arr[${i}]}"
    done
    echo -ne "(default to ${default}. ${items_arr[${default}]})"
    read default
    while [[ ${default} -ge ${i} ]]; do
        echo -ne "${default} is invalid, please re-enter"
        read default
    done
    SELECTED=${items_arr[$default]}
}

function check_network() {
    read_ynflag "is network connected? [y/n]"
    if [ "$YN_FLAG" == "n" ]; then
        exit_script "network is disconnect, abort"
    fi
}

STRARR=""
function to_str_array() {
    local str=$1
    local delimiter=$2
    # separate string into arrays: 
    # IFS = internal field separator, a special context variable
    # <<< to redirect the string as stdin of the command
    IFS=$delimiter read -r -a STRARR <<< $str
}

function check_existence() {
    local arr=$1
    local err_msg=$2
    if [[ ${#arr[@]} -le 0 ]]; then
        exit_script $err_msg
    fi
}

function reboot_with_confirm() {
    read_ynflag "${red_start}reboot now? [y/n]${red_end}"
    if [ "${YN_FLAG}" == "y" ]; then
        sync
        sudo reboot
    fi
}

function install_qt() {
    echo -e "${green_start}install qt${green_end}"
    local qt_path=`ls ${PKG_PATH} | grep -i qt.*\.run`
    select_items $qt_path
    qt_path=${PKG_PATH}/${SELECTED}
    chmod +x $qt_path
    $"$qt_path"
}

function install_typora() {
    echo -e "${green_start}install typora${green_end}"
    wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -
    sudo add-apt-repository 'deb https://typora.io/linux ./'
    sudo apt-get update
    sudo apt-get install typora -y
}

function install_latex() {
    local cfg_dir="${HOME}/.config/texstudio"

    echo -e "${green_start}adding ppa for texstudio${green_end}"
    sudo add-apt-repository ppa:sunderme/texstudio
    sudo apt-get update

    sudo apt install texlive-full
    sudo apt-get install texstudio -y

    read_ynflag "copy the texstudio config to $cfg_dir? (may overwrite current config) [y/n]"
    if [ $YN_FLAG == "y" ]; then
        if [ -d $cfg_dir ]; then
            rm -rf $cfg_dir
        fi
        mkdir -p ${cfg_dir}
        cp -r ./config/texstudio/* ${cfg_dir}/
        echo -e "${green_start}complete texstudio setup${green_end}"
    else
        echo -e "${red_start}texstudio setup NOT complete: need manually copy setup file to ${cfg_dir}${red_end}"
    fi
}

function install_netease_cloud() {
    local netease_pkg=`ls ${PKG_PATH} | grep -i netease-cloud-music.*\.deb`

    read_ynflag "is qt5 installed? [y/n]"
    if [ $YN_FLAG == "n" ]; then
        exit_script "need to install qt5 first, via option -i qt, abort"
    fi

    select_items $netease_pkg
    netease_pkg=${PKG_PATH}/${SELECTED}
    sudo apt install $netease_pkg # install dependencies as well
}

function install_pinyin() {
    echo -e "${green_start}install fcitx${green_end}"
    sudo apt install fcitx -y
    read_ynflag "need to config im-config to use \"fcitx\" as input method configuration, confirm [y/n]"
    `im-config`
    echo -e "${green_start}install google pinyin${green_end}"
    sudo apt install fcitx-googlepinyin -y

    echo -en "${red_start}need to find google pinyin in fcitx after reboot...${red_end}"
    reboot_with_confirm
}

function install_utils() {
    local pkg_list=("typora" "latex" "netease_cloud" "pinyin")
    for pkg in ${pkg_list[@]}; do
        read_ynflag "install ${pkg} [y/n] ? "
        if [ $YN_FLAG == "y" ]; then
            $"install_${pkg}"
        fi
    done
}

function install_personal_env() {
    # change default folder view
    gsettings set org.gnome.nautilus.preferences default-folder-viewer 'list-view'
    # append .bashrc
    cat ./bashrc >> ${HOME}/.bashrc 
    read_ynflag "${green_start}please check ~/.bashrc to verify PATH is set correctly${green_end} [y/n]"
    # config git
    git config --global user.email beihaifusang@gmail.com
    git config --global user.name beihaifusang
    git config --global push.default simple
    # copy vs-code setting
    local code_cfg=$HOMNE/.config/Code/User
    mkdir -p $code_cfg
    ln -s ./config/vscode/settings.json $code_cfg/settings.json
    ln -s ./config/vscode/keybindings.json $code_cfg/keybindings.json

    # install gnome
    sudo apt-get install ubuntu-gnome-desktop -y
    reboot_with_confirm
}

function install_python() {
    read_ynflag "${red_start}has python 3.x installed [y/n]?${red_end}"
    local py_ver="3.6"
    local tmp
    if [ "$YN_FLAG" == "n" ]; then
        read_ynflag "${red_start}install python via apt-get [y/n] ?${red_end}"
        if [ "${YN_FLAG}" == "y" ]; then
            echo -ne "desired python version: (default to 3.6) "
            read tmp
            if [ "$tmp" != "" ]; then
                py_ver=$tmp
            fi
            while [[ ${py_ver//.} -le 35 ]]; do
                echo -ne "should install python>=3.6, please re-enter "
                read tmp
                if [ "$tmp" != "" ]; then
                    py_ver=$tmp
                fi
            done
            echo -e "${green_start}installing py${py_ver}${green_end}"
            sudo apt-get install software-properties-common
            sudo add-apt-repository ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install python${py_ver}
        else
            exit_script "should install python>=3.6 first"
        fi
    fi

    read_ynflag "${red_start}has system default python changed? [y/n]${red_end}"
    if [ "${YN_FLAG}" == "y" ]; then
        read_ynflag "using update-alternatives to manage system python version? [y/n]"
        if [ "${YN_FLAG}" == "y" ]; then
            setup_alternatives_python
        fi
        read_ynflag "using update-alternatives to manage system python ${red_start}3${red_end} version? [y/n]"
        if [ "${YN_FLAG}" == "y" ]; then
            setup_alternatives_python 3
        fi
        read_ynflag "${red_start}want to link previous system python dist-packages to be executed by newly installed python? [y/n]${red_end}"
        if [ "${YN_FLAG}" == "y" ]; then
            link_sys_python_pkgs $py_ver
        fi
    fi

    py_ver=`python -c "import sys; ver=sys.version_info; print(\"%d.%d\" % (ver[0], ver[1]))"`
    if [[ ${py_ver//.} -le 35 ]]; then
        exit_script "should install python>=3.6, but detected python${py_ver}, abort"
    fi

    if [ "`command -v pip3`" == "" ]; then
        echo -e "${green_start}installing pip${green_end}"
        sudo apt-get install python3-pip -y
        python -m pip install --upgrade pip
        python -m pip install setuptools
    fi
    echo -e "install python pkgs"
    sudo apt-get install python3-dev python${py_ver}-dev build-essential
    echo -e "${green_start}installing from ./requirement_py.txt${green_end}"
    python -m pip install -r ./requirement_py.txt

    echo -e "${green_start}python setup finished${green_end}"
    return 0
}

function link_sys_python_pkgs(){
    local ver=${1//.} # remove potential '.'
    local i
    echo $ver
    # set up python3 env after changing sys default python from 3.5 -> 3.6 or greater
    a=$(find /usr/lib/python3/dist-packages -name '*35m*so')
    b=$(echo $a | tr 35m ${ver}m)
    IFS=' ' read -r -a a <<< $a
    IFS=' ' read -r -a b <<< $b
    for (( i=0; i<${#a[@]}; ++i )); do
        echo -e "linking ${a[i]} -> ${b[i]}"
        if [ ! -f ${b[i]} ]; then
            sudo ln -s "${a[i]}" "${b[i]}"
        fi
    done
}

function setup_alternatives_python() {
    local major=$1
    if [ "$major" == "" ]; then
        major="[0-9]"
    fi
    local sys_py=`ls /usr/bin/python${major}.[0-9]`
    local usr_py=`ls /usr/local/bin/python${major}.[0-9]`
    local env_name="python${1}"
    local link="/usr/local/bin/python${1}"
    local tmp
    local i
    local base

    echo -ne "python env name: (default to ${env_name}):"
    read tmp
    if [ "$tmp" != "" ]; then
        env_name=$tmp
    fi

    # sudo update_alternatives --install <link> <name> <path> <priority>
    IFS=' ' read -r -a sys_py <<< $sys_py
    for (( i=0; i<${#sys_py[@]}; ++i )); do
        # echo $link $env_name ${sys_py[$i]} $i
        sudo update-alternatives --install $link $env_name ${sys_py[$i]} $i
    done
    IFS=' ' read -r -a usr_py <<< $usr_py
    base=$i
    for (( i=0; i<${#usr_py[@]}; ++i )); do
        # echo $link $env_name ${usr_py[$i]} $(( $base + $i ))
        sudo update-alternatives --install $link $env_name ${usr_py[$i]} $(( $base + $i ))
    done
}

function configure_env() {
    echo -e "${green_start}using packages in [${PKG_PATH}]${green_end}"
    if [ ! -d ${PKG_PATH} ]; then
        exit_script "input packages: PKG_PATH not valid"
    fi
    return 0
}

function append_env_var() {
    local var=$1
    local dir=$2
    local file=$3

    vdata=`grep -E "export ${var}=" ${file}`
    sdata=`echo ${vdata} | grep ${dir}`
    if [ "${vdata}" == "" -o "${sdata}" == "" ]; then
        echo -e "export ${var}=${dir}:\${${var}}" >> ${file}
    fi
}

function install_nvidia() {
    local d_name=${PKG_PATH}/nvidia
    local p_name=`ls ${d_name} | grep -i NVIDIA.*run` # find all .run installer

    if [[ ${#p_name[@]} -le 0 ]]; then
        exit_script "no available run-file under ${d_name}, abort"
        return 1
    fi

    echo -e "${green_start}install nvidia driver${green_end}"
    select_items ${p_name}
    p_name=$SELECTED

    read_ynflag "${red_start} press ctrl + alt + F1 to switch mode [y/n]?${red_end}"
    if [ "${YN_FLAG}" != "y" ]; then
        return 1
    fi

    # configure nouveau
    local b_file=/etc/modprobe.d/blacklist.conf
    if [ -z "`grep -E 'blacklist nouveau' ${b_file}`" ]; then
        echo -e "${green_start}configure blacklist.conf${green_end}"
        sudo bash -c "echo -e 'blacklist nouveau\noptions nouveau modeset=0' >> ${b_file}"
        sudo update-initramfs -u # possible found missing i915 moduls (yet not important)
        echo -en "${red_start} need to restart to finalize backlisting nouveau...${red_end}"
        reboot_with_confirm
    fi

    sudo service lightdm stop
    sudo ${d_name}/${p_name} -no-x-check -no-nouveau-check -no-opengl-files # dkms

    reboot_with_confirm
    return 0
}

function install_cuda() {
    local cuda_dir=${PKG_PATH}/nvidia
    local cuda_dep=`ls ${cuda_dir} | grep -i cuda-.*`
    select_items ${cuda_dep}
    cuda_dep=$SELECTED

    sudo dpkg -i "${cuda_dir}/${cuda_dep}"
    local cuda_ver=`ls /var/ | grep -i cuda-.*`
    select_items ${cuda_ver}
    cuda_ver=$SELECTED
    sudo apt-key add /var/${cuda_ver}/7fa2af80.pub

    # Meta Package		        Purpose
    # cuda			            Installs all CUDA Toolkit and Driver packages. Handles upgrading to the next version of the cuda package when it's released.
    # cuda-10-1		            Installs all CUDA Toolkit and Driver packages. Remains at version 10.1 until an additional version of CUDA is installed.
    # cuda-toolkit-10-1	        Installs all CUDA Toolkit packages required to develop CUDA applications. Does not include the driver.
    # cuda-tools-10-1		    Installs all CUDA command line and visual tools.
    # cuda-runtime-10-1	        Installs all CUDA Toolkit packages required to run CUDA applications, as well as the Driver packages.
    # cuda-compiler-10-1	    Installs all CUDA compiler packages.
    # cuda-libraries-10-1	    Installs all runtime CUDA Library packages.
    # cuda-libraries-dev-10-1	Installs all development CUDA Library packages.
    # cuda-drivers		        Installs all Driver packages. Handles upgrading to the next version of the Driver packages when they're released.
    sudo apt-get update
    sudo apt-get install cuda-toolkit-10-1 -y

    reboot_with_confirm
    return 0
}

function install_cudnn() {
    # fetch system cuda version
    if [[ `which nvcc` == "" ]]; then
        exit_script "nvcc not found, should install cuda first, abort"
    fi
    to_str_array "`nvcc --version`" ","
    to_str_array "${STRARR[-2]}" " "
    local cuda_ver=${STRARR[-1]}

    local cudnn_dir=${PKG_PATH}/nvidia
    local dir_p=`find ${cudnn_dir} -type d -name "*cudnn*"` # all available dir
    select_items ${dir_p} # select a dir
    dir_p=$SELECTED
    local deb_list=`ls $dir_p/*cuda${cuda_ver}*` # get .dep for the chosen cudnn version (that matches cuda ver)
    if [[ ${#deb_list[*]} != 3 ]]; then
        exit_script "please check deb files that matches cuda version ${cuda_ver}"
    fi
    for i in ${deb_list[@]}; do # install
        sudo dpkg -i ${dir_p}/${i}
    done
    reboot_with_confirm
}

function install_torch() {
    local torch_src=$1
    local torch_dir=torch-1.3.0
    local torch_tar=${torch_dir}.tar.gz
    local torch_del=${home_path}/software/torch*
    local torch_path=${home_path}/software/${torch_dir}

    if [ ${torch_src} == "ftp" ]; then
        check_network
        rm -rf ./${torch_tar} ${torch_del}
        mkdir -p ${torch_path}

        wget -nH ftp://${ftp_server}:${ftp_port}${ftp_path}/${torch_tar} --ftp-user=${ftp_user} --ftp-password=${ftp_passwd} -O ${torch_tar}
        if [ -s ./${torch_tar} ]; then
            tar xzf ./${torch_tar} -C ${torch_path}
            rm -rf ./${torch_tar}
        else
            echo -e "${red_start}failed to download library [${torch_tar}]${red_end}\n"
            rm -rf ./${torch_tar}
            return 1
        fi
    else
        if [ -f "${torch_src}" ]; then
            rm -rf ${torch_del}
            mkdir -p ${torch_path}
            tar xzf ${torch_src} -C ${torch_path}
        else
            echo -e "${red_start}${torch_src} is not found${red_end}\n"
            return 1
        fi
    fi

    create_env_file
    local torch_lib=${torch_path}/lib
    append_env_var LD_LIBRARY_PATH ${torch_lib} ${efile}
    return 0
}

function install_input_list() {
    for item in $@; do
        echo -e "${green_start}install ${item} ...${green_end}"
        sudo apt-get install ${item} -y
        echo -e "${green_start}install ${item} completed!${green_end}\n"
    done
}

function install_general() { # install general apt pkgs
    local ubuntu_ver=`lsb_release -r | awk '{print $2}'`
    echo -e "${green_start}ubuntu version is ${ubuntu_ver}${green_end}"

    check_network    
    if [ "${ubuntu_ver}" == "16.04" ]; then
        # install list
        local install_item=(
            cmake
            curl
            dkms
            filezilla
            git
            gnuplot
            nethogs
            openconnect
            openssh-server
            libssl-dev 
            libncurses5-dev 
            libsqlite3-dev 
            libreadline-dev 
            libtk8.5 
            libgdm-dev 
            libdb4o-cil-dev 
            libpcap-dev
            vim
            libglib2.0-bin
            htop
        )

        # update
        sudo apt-get update -y

        # install
        install_input_list ${install_item[*]}
    else
        echo -e "${red_start}unsupport ubuntu version: ${ubuntu_ver}${red_end}"
    fi
    return 0
}


# execute environment
if [ $# == 0 ]; then
    help_content
    exit 1
fi

# -o: start listing short args
# --long: start listing long args 
# $@: fetch stdin string into an array
ARGS=`getopt -o p:i: --long path:,install: -- "$@"`
eval set -- "${ARGS}" # re-allocate parsed args (key-val) to $1, $2, ...

PKG_PATH="./pkgs" # solve first-class citizen
while true; do
    case ${1} in
        -p|--path)
            PKG_PATH=$(realpath $2)
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

eval set -- "${ARGS}" 
while true; do
    case ${1} in
        -p|--path)
            shift 2
            ;;
        -i|--install)
            list=$2
            for i in ${list[@]}; do
                case $i in
                    nvidia)
                        install_nvidia
                        ;;
                    cuda)
                        install_cuda
                        ;;
                    cudnn)
                        install_cudnn
                        ;;
                    python)
                        install_python
                        ;;
                    qt)
                        install_qt
                        ;;
                    pinyin)
                        install_pinyin
                        ;;
                    personal_env)
                        install_personal_env
                        ;;
                    general)
                        install_general
                        ;;
                    utils)
                        install_utils
                        ;;
                    tex)
                        install_latex
                        ;;
                    all)
                        pkg_list=("general" "python" "qt" "utils" "nvidia" "cuda" "cudnn" "personal_env")
                        for pkg in ${pkg_list[@]}; do
                            read_ynflag "install ${pkg} [y/n] ? "
                            if [ $YN_FLAG == "y" ]; then
                                $"install_${pkg}"
                            fi
                        done
                        ;;
                    --)
                        break
                        ;;
                    *)
                        help_content
                        exit_script "invalid args, abort"
                        ;;
                esac
            done
            shift $(( ${#list[@]} + 1 ))
            ;;
        --)
            break
            ;;
        *)
            help_content
            exit_script "invalid args, abort"
            ;;
    esac
done

echo -e "${green_start}install env successfully${green_end}"
exit 0
