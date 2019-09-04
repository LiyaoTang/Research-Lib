#pragma once
#include <stdint.h>
#include <chrono>
#include <iostream>

namespace time {

enum TimeType {
    SECOND,
    MILLISECOND,
    MICROSECOND,
    NANOSECOND
};

class TimeStamp {
private:
    class TimeScale {
    public:
        TimeScale(TimeType time_type) {
            switch (time_type) {
                case SECOND:
                    _scale = 1.0;
                    break;
                case MILLISECOND:
                    _scale = 1e-3;
                    break;
                case MICROSECOND:
                    _scale = 1e-6;
                    break;
                case NANOSECOND:
                    _scale = 1e-9;
                    break;
            }
        }

        operator double() {
            return _scale;
        }

    private:
        double _scale;
    };

public:
    TimeStamp();

    TimeStamp(int64_t time_stamp, TimeType time_type = NANOSECOND);

    virtual ~TimeStamp();

    TimeStamp operator-(const TimeStamp &rhs) const;

    TimeStamp operator+(const TimeStamp &rhs) const;

    TimeStamp operator/(const int64_t &s) const;

    TimeStamp &operator+=(const TimeStamp &rhs);

    TimeStamp &operator-=(const TimeStamp &rhs);

    TimeStamp &operator/=(const int64_t &s);

    bool operator==(const TimeStamp &rhs) const;

    bool operator!=(const TimeStamp &rhs) const;

    bool operator>(const TimeStamp &rhs) const;

    bool operator<(const TimeStamp &rhs) const;

    bool operator>=(const TimeStamp &rhs) const;

    bool operator<=(const TimeStamp &rhs) const;

    int64_t &operator()();

    TimeStamp abs() const;

    double to_sec() const;

    static TimeStamp now(TimeType time_type = NANOSECOND);

    static TimeStamp from_string(const std::string time, const std::string &type = "sec");

    static TimeStamp from_date(const std::string date);  //2017-09-28 05:22:52.884694
    static TimeStamp from_sec_nsec(const std::string time);

    static TimeStamp from_nsec(const std::string time);

    friend std::ostream &operator<<(std::ostream &out, const TimeStamp &rhs);

    double get_double() const;

    int64_t get_int64() const;

    uint32_t get_sec() const;

    uint32_t get_nsec() const;

    TimeType get_timetype() const { return _time_type; }

private:
    int64_t _time_stamp;
    TimeType _time_type;
};

}  // namespace time
