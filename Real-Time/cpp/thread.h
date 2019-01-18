#ifndef THREAD_H
#define THREAD_H

#include <string>
#include <pthread.h>
#include "noncopyable.h"


class Thread {
public:
    Thread(bool joinable = false,
            const std::string &name = "Thread",
            const int cancel_type = PTHREAD_CANCEL_DEFERRED);

    void start();
    void join();

    pthread_t tid() const;
    bool is_alive() const;

    const bool joinable;
    const std::string thread_name;

protected:
    virtual void run() = 0; // pure virtual => abstract class
    static void *thread_runner(void *arg);

    pthread_t _tid;
    bool _started;
    const int _cancel_type;
    
private:
    DISALLOW_COPY_AND_ASSIGN(Thread);
};

#endif // THREAD_H
