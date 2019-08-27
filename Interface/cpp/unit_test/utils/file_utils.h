#pragma once
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

bool is_file_exist(const std::string &filename);

template <typename T>
std::string number2string(T num, int len, char full) {
    std::stringstream ss;
    if (len > 0) {
        ss << std::setw(len) << std::setfill(full);
    }

    ss << num;
    return ss.str();
}

template <typename T>
T string2number(const std::string &str, T defalut_value) {
    std::stringstream ss(str);

    T value = defalut_value;
    ss >> value;
    return value;
}

bool sys_cmd(const std::string &cmd);

bool sys_cmd(const std::string &cmd, std::vector<std::string> &lines);

bool get_list_files(
    const std::string &path,
    std::vector<std::string> &filenames,
    const std::string &exten = "",
    bool is_full_path        = true);

bool delete_file(const std::string &file);

bool delete_dir_all_file(const std::string &path);

bool create_folder(const std::string &dir);

std::string get_current_exe_dir();
