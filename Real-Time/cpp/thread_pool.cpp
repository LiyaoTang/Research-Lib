#include "thread_pool.h"

using std::vector;
using google::protobuf::Closure;

ThreadPool::ThreadPool(int worker_num) :
        worker_num(worker_num),
        available_worker_num(worker_num),
        _task_queue(worker_num),
        _started(false) {

    _workers.reserve(this->worker_num);
    for (int idx = 0; idx < this->worker_num; idx++) {
        ThreadWorker* worker = new ThreadWorker(this);
        _workers.push_back(worker);
    }
}

ThreadPool::~ThreadPool() {
    if (!this->_started) {
        return;
    }
    for (int idx = 0; idx < this->worker_num; idx++) {
        this->add(NULL);
    }
    for (int idx = 0; idx < this->worker_num; idx++) {
        this->_workers[idx]->join();
        delete this->_workers[idx];
    }
}

void ThreadPool::start() {
    for (int idx = 0; idx < this->worker_num; idx++) {
        this->_workers[idx]->start(); // start the thread
    }
    this->_started = true;
}

void ThreadPool::add(Closure* closure) {
    this->_task_queue.push(closure);
}

void ThreadPool::add(const vector<Closure*>& closures) {
    for (size_t idx = 0; idx < closures.size(); idx++) {
        add(closures[idx]);
    }
}

void ThreadWorker::run() {
    while (true) {
        Closure *closure = NULL;
        this->_thread_pool->_task_queue.pop(&closure);
        if (closure == NULL) {
            break;
        }
        (this->_thread_pool->available_worker_num)--;
        closure->Run();
        (this->_thread_pool->available_worker_num)++;
    }
}
