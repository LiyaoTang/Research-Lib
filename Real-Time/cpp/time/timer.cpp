#include "timer.h"

namespace time {

Timer::Timer(const std::string &name) : _start_time(0), _total(0), _name(name) {
    start();
}

Timer::~Timer() {
}

/**
 * timer start
 */
void Timer::start() {
    _total      = TimeStamp(0);
    _start_time = TimeStamp::now();
}

/**
 * get the time between start() and stop().
 */
double Timer::read() {
    _total = (TimeStamp::now() - _start_time);
    return _total.to_sec();
}

void Timer::print(const std::string &addtion, bool restart) {
    read();
    std::cout << _name << " " << addtion << ": " << _total.to_sec() << std::endl;
    if (restart) {
        start();
    }
}

std::ostream &operator<<(std::ostream &out, Timer &timer) {
    out << timer._name << " " << timer.read();
    return out;
}

}  // namespace time