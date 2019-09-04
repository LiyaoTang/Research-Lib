#pragma once
#include <iostream>
#include "time_stamp.h"

namespace time {

class Timer {
public:
    Timer(const std::string &name = "");

    ~Timer();

    void start();

    double read();

    void print(const std::string &addtion = "", bool restart = false);

    friend std::ostream &operator<<(std::ostream &out, Timer &timer);

private:
    TimeStamp _start_time;
    TimeStamp _total;

    std::string _name;
};

}  // namespace time
