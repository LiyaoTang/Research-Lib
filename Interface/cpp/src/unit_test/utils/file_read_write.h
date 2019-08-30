#pragma once
#include <string>
#include <vector>

std::string trim(const std::string &s, char t = 'z');

void split(
    const std::string &line,
    std::vector<std::string> &array,
    char separate = ' ');

bool load_text_to_string(
    const std::string &filename,
    std::vector<std::string> &lines,
    size_t offset = 0,
    char common   = '#');

bool save_string_to_text(const std::string &filename,
                         const std::vector<std::string> &lines,
                         bool isappend = false);

bool save_string_to_text(
    const std::string &filename,
    const std::string &lines,
    bool isappend = false);

void string_replace(
    std::string &str,
    const std::string &find_str,
    const std::string &replace_str);

bool write_to_file(const std::string &filename, const std::string &values, bool is_append);
