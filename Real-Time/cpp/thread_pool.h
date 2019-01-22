#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <google/protobuf/stubs/common.h>
#include <pthread.h>
#include <vector>

#include "concurrent_base.h"
#include "concurrent_queue.h"
#include "noncopyable.h"
#include "thread.h"

class ThreadWorker;

class ThreadPool {
public:
    explicit ThreadPool(int num_workers);
    ~ThreadPool();

    void start();

    void add(google::protobuf::Closure* closure);
    void add(const std::vector<google::protobuf::Closure*>& closures);

    const int worker_num = 0;
    ConcurrentBase<int> available_worker_num = 0;

private:
    bool _started = false;
    std::vector<ThreadWorker*> _workers;
    FixedSizeConQueue<google::protobuf::Closure*> _task_queue;

    DISALLOW_COPY_AND_ASSIGN(ThreadPool);
    friend class ThreadWorker;
};

class ThreadWorker : public Thread {
public:
    explicit ThreadWorker(ThreadPool* thread_pool) : Thread(true, "ThreadWorker"),
                                                     _thread_pool(thread_pool) {}

    virtual ~ThreadWorker() {}

protected:
    virtual void run() override;

private:
    DISALLOW_COPY_AND_ASSIGN(ThreadWorker);
    ThreadPool* _thread_pool;
};

#endif  // THREAD_POOL_H
