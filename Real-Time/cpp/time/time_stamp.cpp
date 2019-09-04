#include "time_stamp.h"
#include <string.h>
#include <cassert>
#include <cmath>
#include <sstream>

namespace time {

TimeStamp::TimeStamp() : _time_stamp(0),
                         _time_type(NANOSECOND) {
}

TimeStamp::TimeStamp(int64_t time_stamp, TimeType time_type) : _time_stamp(time_stamp),
                                                               _time_type(time_type) {
}

TimeStamp::~TimeStamp() {
}

TimeStamp TimeStamp::operator-(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return TimeStamp(_time_stamp - rhs.get_int64(), _time_type);
}

TimeStamp TimeStamp::operator+(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return TimeStamp(_time_stamp + rhs.get_int64(), _time_type);
}

TimeStamp TimeStamp::operator/(const int64_t &s) const {
    return TimeStamp(_time_stamp / s, _time_type);
}

TimeStamp &TimeStamp::operator+=(const TimeStamp &rhs) {
    assert(_time_type == rhs._time_type);
    _time_stamp += rhs.get_int64();
    return *this;
}

TimeStamp &TimeStamp::operator-=(const TimeStamp &rhs) {
    assert(_time_type == rhs._time_type);
    _time_stamp -= rhs.get_int64();
    return *this;
}

TimeStamp &TimeStamp::operator/=(const int64_t &s) {
    _time_stamp /= s;
    return *this;
}

bool TimeStamp::operator==(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp == rhs.get_int64();
}

bool TimeStamp::operator!=(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp != rhs.get_int64();
}

bool TimeStamp::operator>(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp > rhs.get_int64();
}

bool TimeStamp::operator<(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp < rhs.get_int64();
}

bool TimeStamp::operator>=(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp >= rhs.get_int64();
}

bool TimeStamp::operator<=(const TimeStamp &rhs) const {
    assert(_time_type == rhs._time_type);
    return _time_stamp <= rhs.get_int64();
}

int64_t &TimeStamp::operator()() {
    return _time_stamp;
}

TimeStamp TimeStamp::abs() const {
    return TimeStamp(std::abs(get_int64()), _time_type);
}

double TimeStamp::to_sec() const {
    return _time_stamp * double(TimeScale(_time_type));
}

TimeStamp TimeStamp::now(TimeType time_type) {
    int64_t t;
    auto now_t = std::chrono::system_clock::now().time_since_epoch();
    switch (time_type) {
        case SECOND:
            t = now_t / std::chrono::seconds(1);
            break;
        case MILLISECOND:
            t = now_t / std::chrono::milliseconds(1);
            break;
        case MICROSECOND:
            t = now_t / std::chrono::microseconds(1);
            break;
        case NANOSECOND:
            t = now_t / std::chrono::nanoseconds(1);
            break;
        default:
            t         = now_t / std::chrono::nanoseconds(1);
            time_type = NANOSECOND;
            break;
    }

    return TimeStamp(t, time_type);
}

TimeStamp TimeStamp::from_string(const std::string time, const std::string &type) {
    if (type == "sec") {
        return from_nsec(time);
    } else if (type == "date") {
        return from_date(time);
    } else if (type == "secnsec") {
        return from_sec_nsec(time);
    } else {
        //pass
    }

    return TimeStamp(0);
}

TimeStamp TimeStamp::from_date(const std::string date) {
    struct tm stm;
    int ims              = 0;
    const char *str_time = date.data();

    memset(&stm, 0, sizeof(stm));
    stm.tm_year = atoi(str_time) - 1900;
    stm.tm_mon  = atoi(str_time + 5) - 1;
    stm.tm_mday = atoi(str_time + 8);
    stm.tm_hour = atoi(str_time + 11);
    stm.tm_min  = atoi(str_time + 14);
    stm.tm_sec  = atoi(str_time + 17);
    ims         = atoi(str_time + 20);

    return TimeStamp(int64_t(mktime(&stm) * 1e9) + ims * 1000);
}

TimeStamp TimeStamp::from_sec_nsec(const std::string time) {
    std::string::size_type idx = time.find_first_of('.');
    if (std::string::npos == idx) {
        return from_nsec(time);
    } else {
        int64_t second     = 0;
        int64_t nsecond    = 0;
        int64_t time_stamp = 0;
        std::stringstream ss(time.substr(0, idx));
        ss >> second;

        std::string nsecond_str = time.substr(idx + 1, time.length());
        ss.str("");
        ss.clear();
        ss << nsecond_str;
        ss >> nsecond;

        time_stamp = second * 1e9;
        time_stamp += int64_t(nsecond * pow(10.0, 9 - nsecond_str.length()));
        return TimeStamp(time_stamp);
    }
}

TimeStamp TimeStamp::from_nsec(const std::string time) {
    std::stringstream ss(time);
    int64_t time_stamp;
    ss >> time_stamp;
    return TimeStamp(time_stamp);
}

std::ostream &operator<<(std::ostream &out, const TimeStamp &rhs) {
    return out << rhs.get_int64();
}

double TimeStamp::get_double() const {
    double t = (double)_time_stamp;
    return t;
}

int64_t TimeStamp::get_int64() const {
    return _time_stamp;
}

uint32_t TimeStamp::get_sec() const {
    return uint32_t(_time_stamp * TimeScale(_time_type));
}

uint32_t TimeStamp::get_nsec() const {
    return uint32_t(_time_stamp % int64_t(1.0 / TimeScale(_time_type)) * TimeScale(_time_type) * 1e9);
}

}  // namespace time
