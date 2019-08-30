#include "file_utils.h"
#include <unistd.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "file_read_write.h"

bool is_file_exist(const std::string &filename) {
    std::fstream file_reader(filename, std::ios::in);
    if (!file_reader.is_open()) {
        std::cout << "[warning]: the file[" << filename << "] is not existed!" << std::endl;
        return false;
    }

    file_reader.close();

    return true;
}

bool sys_cmd(const std::string &cmd) {
    std::vector<std::string> lines;

    return sys_cmd(cmd, lines);
}

bool sys_cmd(const std::string &cmd, std::vector<std::string> &lines) {
    FILE *fp = popen(cmd.c_str(), "r");

    lines.clear();

    char buffer[512];
    while (fgets(buffer, sizeof(buffer), fp)) {
        std::string line(buffer);
        line = trim(line);
        lines.push_back(line);
        //        std::cout << line << std::endl;
    }

    fclose(fp);

    //    if (lines.empty()) {
    //        std::cout << "[warning]: " << cmd  << " fail" << std::endl;
    //        return false;
    //    }

    return true;
}

/*
 * exten file filter by grep
 * is_full_path: whether return full path for each file
 */
bool get_list_files(
    const std::string &path,
    std::vector<std::string> &filenames,
    const std::string &exten,
    bool is_full_path) {
    std::string cmd = "cd " + path + " &&ls |grep " + exten;

    if (exten.empty()) {
        cmd = "cd " + path + " &&ls";
    }

    if (!sys_cmd(cmd, filenames)) {
        return false;
    }

    if (is_full_path) {
        for (size_t i = 0; i < filenames.size(); i++) {
            filenames[i] = path + "/" + filenames[i];
        }
    }

    return true;
}

bool delete_file(const std::string &file) {
    std::string cmd = "rm " + file;
    return sys_cmd(cmd);
}

bool delete_dir_all_file(const std::string &path) {
    std::vector<std::string> filenames;
    get_list_files(path, filenames, "", true);

    for (size_t i = 0; i < filenames.size(); ++i) {
        delete_file(filenames[i]);
    }

    return true;
}

bool create_folder(const std::string &dir) {
    std::string cmd = "mkdir  " + dir;
    return sys_cmd(cmd);
}

std::string get_current_exe_dir() {
    // get current work dir
    char buf[1024];
    int count = readlink("/proc/self/exe", buf, 1024);
    if (count < 0 || count >= 1024) {
        std::cout << "Get current work dir failure" << std::endl;
    }
    buf[count] = '\0';
    std::string raw_dir(buf);
    return raw_dir;
}
