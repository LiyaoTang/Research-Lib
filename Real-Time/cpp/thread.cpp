#include <assert.h>
#include <glog/logging.h>
#include <signal.h>

#include "thread.h"

Thread::Thread(bool joinable = false,
               const std::string& name = "Thread",
               const int cancel_type = PTHREAD_CANCEL_DEFERRED) : _tid(0),
                                                                  _started(false),
                                                                  joinable(joinable),
                                                                  thread_name(name),
                                                                  _cancel_type(cancel_type) {}

void Thread::start() {
    pthread_attr_t attr;
    CHECK_EQ(pthread_attr_init(&attr), 0);

    CHECK_EQ(
        pthread_attr_setdetachstate(
            &attr,
            this->joinable ? PTHREAD_CREATE_JOINABLE : PTHREAD_CREATE_DETACHED),
        0);
    CHECK_EQ(pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL), 0);
    // cautious to use PTHREAD_CANCEL_ASYNCHRONOUS
    CHECK_EQ(pthread_setcanceltype(this->_cancel_type, NULL), 0);  // <-- invalid cancel_type trigger error here

    // pass in the object as well => derived class can define task specific variables & use in its thread
    // "reinterpret_cast" to bypass/ignore intermediate check/result
    int result = pthread_create(&(this->_tid), &attr, &(this->thread_runner), reinterpret_cast<void*>(this));
    CHECK_EQ(result, 0) << "Could not create thread (" << result << ")";

    CHECK_EQ(pthread_attr_destroy(&attr), 0);
    _started = true;
}

void Thread::join() {
    CHECK(this->joinable) << "Thread is not joinable";
    int result = pthread_join(this->_tid, NULL);
    CHECK_EQ(result, 0) << "Could not join thread (" << _tid << ", " << this->thread_name << ")";
    this->_tid = 0;
}

pthread_t Thread::tid() const {
    return this->_tid;
}

bool Thread::is_alive() const {
    if (this->_tid == 0) {
        return false;
    }
    // no signal sent, just check existence for thread
    int ret = pthread_kill(this->_tid, 0);
    if (ret == ESRCH) {
        return false;
    }
    if (ret == EINVAL) {
        LOG(WARNING) << "Invalid singal.";
        return false;
    }
    return true;
}

void* Thread::thread_runner(void* arg) {
    // inside a separate thread now
    Thread* t = reinterpret_cast<Thread*>(arg);  // convert back to current object
    t->run();                                    // actual control flow
    return NULL;
}
