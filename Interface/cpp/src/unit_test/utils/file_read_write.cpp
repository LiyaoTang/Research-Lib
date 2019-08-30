#include "file_read_write.h"
#include <fstream>
#include <iostream>
#include "file_utils.h"

std::string trim(const std::string &s, char t) {
    if (s.empty()) {
        return s;
    }

    int begin_idx = 0;
    int end_idx = s.length() - 1;
    if (t != 'z') {
        while (begin_idx <= end_idx && s[begin_idx] == t) { begin_idx++; }
        while (end_idx >= begin_idx && s[end_idx] == t) { end_idx--; }
    } else {
        while (begin_idx <= end_idx
               && (s[begin_idx] == '\r'
                   || s[begin_idx] == '\n'
                   || s[begin_idx] == '\t'
                   || s[begin_idx] == ' ')) {
            begin_idx++;
        }
        while (end_idx >= begin_idx
               && (s[end_idx] == '\r'
                   || s[end_idx] == '\n'
                   || s[end_idx] == '\t'
                   || s[end_idx] == ' ')) {
            end_idx--;
        }
    }

    return s.substr(begin_idx, end_idx + 1);
}

void split(const std::string &line, std::vector <std::string> &array, char separate) {
    if (line.empty()) {
        std::cerr << "the line is empty, please check it!" << std::endl;
        return;
    }

    std::string::size_type start = 0;
    std::string sub_string;

    array.clear();
    std::string::size_type index = line.find_first_of(separate, start);

//    if (index == std::string::npos) {
//        array.push_back(line);
//        return;
//    }

    while (index != std::string::npos) {
        sub_string = line.substr(start, index - start);
        if (sub_string.length()) {
            array.push_back(sub_string);
        }
        start = index + 1;
        index = line.find_first_of(separate, start);
    }

    if (start != line.length()) {
        sub_string = line.substr(start);
        if (sub_string.length()) {
            array.push_back(sub_string);
        }
    }
}

bool load_text_to_string(
        const std::string &filename,
        std::vector <std::string> &lines,
        size_t offset,
        char common) {

    if (!is_file_exist(filename)) {
        return false;
    }

    std::fstream file_reader(filename, std::ios::in);

    lines.clear();
    std::string data_line;
    size_t i_counter = 0;
    while (getline(file_reader, data_line)) {
        if (data_line[0] == common) {
            continue;
        }
        if (++i_counter < offset) {
            continue;
        }
        lines.push_back(trim(data_line));
    }

    return true;
}

bool save_string_to_text(const std::string &filename, const std::vector <std::string> &lines, bool isappend) {
    std::fstream file_writer;

    if (isappend) {
        file_writer.open(filename, std::ios::out | std::ios::app);
    } else {
        file_writer.open(filename, std::ios::out);
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        file_writer << lines[i] << std::endl;
    }

    file_writer.close();
    return true;
}

bool save_string_to_text(const std::string &filename, const std::string &lines, bool isappend) {
    std::fstream file_writer;

    if (isappend) {
        file_writer.open(filename, std::ios::out | std::ios::app);
    } else {
        file_writer.open(filename, std::ios::out);
    }

    file_writer << lines;

    file_writer.close();
    return true;

}

//原字符串，要替换的字符串，替换为什么字符串
void string_replace(std::string &str, const std::string &find_str, const std::string &replace_str) {
    std::string::size_type pos = 0;//位置
    std::string::size_type replace_len = replace_str.size();//要替换的字符串大小
    std::string::size_type find_len = find_str.size();//目标字符串大小
    while ((pos = str.find(find_str, pos)) != std::string::npos) {
        str.replace(pos, find_len, replace_str);
        pos += replace_len;
    }
}

bool write_to_file(const std::string &filename, const std::string &values, bool is_append) {
    return save_string_to_text(filename, values, is_append);
}