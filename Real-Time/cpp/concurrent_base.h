#ifndef CONCURRENT_BASE_TYPE_H
#define CONCURRENT_BASE_TYPE_H

#include <pthread.h>
#include "noncopyable.h"

template <typename T>
class ConcurrentBase {
public:
    ConcurrentBase(T val) : val(val) {
        // must provide default value
        pthread_rwlock_init(this->rwlock);
    }

    bool operator==(const T v) {
        pthread_rwlock_rdlock(this->rwlock);
        bool rst = (this->val == v);
        pthread_rwlock_unlock(this->rwlock);
        return rst;
    }
    bool operator!=(const T v) {
        return !(*this == v);
    }
    T operator+(const T v) {
        pthread_rwlock_rdlock(this->rwlock);
        T rst = this->val + v;
        pthread_rwlock_unlock(this->rwlock);
        return rst
    }
    T operator-(const T v) {
        pthread_rwlock_rdlock(this->rwlock);
        T rst = this->val - v;
        pthread_rwlock_unlock(this->rwlock);
        return rst
    }
    T operator*(const T v) {
        pthread_rwlock_rdlock(this->rwlock);
        T rst = this->val * v;
        pthread_rwlock_unlock(this->rwlock);
        return rst
    }
    T operator/(const T v) {
        pthread_rwlock_rdlock(this->rwlock);
        T rst = this->val / v;
        pthread_rwlock_unlock(this->rwlock);
        return rst
    }

    const T operator=(T& lv, const ConcurrentBase& rv) {
        // allow to access the value by assignment and pass as constant down "=" chain
        pthread_rwlock_rdlock(rv->rwlock);
        lv = rv->val;
        pthread_rwlock_unlock(rv->rwlock);
        return lv
    }
    const T operator=(const T v) {
        // allow to pass the constant down "=" chain
        pthread_rwlock_wrlock(this->rwlock);
        this->val = v;
        pthread_rwlock_unlock(this->rwlock);
        return v;
    }

    void operator+=(T v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val += rv;
        pthread_rwlock_unlock(this->rwlock);
    }
    void operator-=(T v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val -= rv;
        pthread_rwlock_unlock(this->rwlock);
    }
    void operator*=(T v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val *= rv;
        pthread_rwlock_unlock(this->rwlock);
    }
    void operator/=(T v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val /= rv;
        pthread_rwlock_unlock(this->rwlock);
    }
    void operator%=(T v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val %= rv;
        pthread_rwlock_unlock(this->rwlock);
    }

    T operator++() {
        pthread_rwlock_wrlock(this->rwlock);
        ++this->val;
        pthread_rwlock_unlock(this->rwlock);
    }
    T operator--() {
        pthread_rwlock_wrlock(this->rwlock);
        --this->val;
        pthread_rwlock_unlock(this->rwlock);
    }
    T operator++(int v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val++;
        pthread_rwlock_unlock(this->rwlock);
    }
    T operator--(int v) {
        pthread_rwlock_wrlock(this->rwlock);
        this->val--;
        pthread_rwlock_unlock(this->rwlock);
    }

    explicit ~ConcurrentBase() {
        delete this->val;
        pthread_rwlock_destroy(rwlock);
    }

    T val;
    pthread_rwlock_t rwlock;

private:
    DISALLOW_COPY(ConcurrentBase);
    int b = 0;
};

#endif  // CONCURRENT_BASE_TYPE_H