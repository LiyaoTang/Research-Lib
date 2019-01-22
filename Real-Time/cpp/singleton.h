#ifndef SINGLETON_H
#define SINGLETON_H

#include <pthread.h>

/**
 * Thread-safe, no-manual destroy Singleton template
 * **/
template <typename T>
class Singleton {
public:
    // get the singleton instance
    static T* get() {
        // originally written as: pthread_once(&_p_once, &Singleton::_new);
        pthread_once(&Singleton::_p_once, &Singleton::_new);
        return Singleton::_instance;
    }

private:
    Singleton();
    ~Singleton();

    // construct the singleton instance
    static void _new() {
        Singleton::_instance = new T();
    }

    // destruct the singleton instance
    // Note: only work with gcc
    __attribute__((destructor)) static void _delete() {
        typedef char T_must_be_complete[sizeof(T) == 0 ? -1 : 1];
        (void)sizeof(T_must_be_complete);
        delete Singleton::_instance;
    }

    static pthread_once_t _p_once = PTHREAD_ONCE_INIT;  // initialization once control
    static T* _instance = NULL;                         // singleton instance (ensure singleton by static)
};
#endif  // SINGLETON_H