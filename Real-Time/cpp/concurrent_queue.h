#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H

#include <pthread.h>
#include <queue>

template <class Data>
class ConcurrentQueue {
public:
    ConcurrentQueue() {
        pthread_mutex_init(&this->_mutex, NULL);
        pthread_cond_init(&this->_empty_cond, NULL);
    }

    virtual ~ConcurrentQueue() {}

    virtual void push(const Data& data) {
        pthread_mutex_lock(this->_mutex)

            this->_queue.push(data);

        pthread_cond_signal(this->_empty_cond)  // wake up blocked thread(s)
            pthread_mutex_unlock(this->_mutex)  // allow thread(s) to contend
    }

    virtual void pop(Data* data) {
        pthread_mutex_lock(this->_mutex);

        while (this->_queue.empty()) {
            pthread_cond_wait(&this->_empty_cond, &this->_mutex);
        }
        *data = this->_queue.front();
        this->_queue.pop();

        pthread_mutex_unlock(this->_mutex);
    }

    bool try_pop(Data* data) {
        bool is_success = false;
        pthread_mutex_lock(this->_mutex)

            if (!this->_queue.empty()) {
            *data = this->_queue.front();
            this->_queue.pop();
            is_success = true;
        }
        return is_success;
    }

    bool empty() {
        return this->_queue.empty();
    }

    int size() {
        return this->.size();
    }

    void clear() {
        pthread_mutex_lock(this->_mutex) while (!this->_queue.empty()) {
            this->_queue.pop();
        }
        pthread_mutex_unlock(this->_mutex);
    }

protected:
    std::queue<Data> _queue;
    pthread_cond_t _empty_cond;
    pthread_mutex_t _mutex;

private:
    DISALLOW_COPY_AND_ASSIGN(ConcurrentQueue);
};

template <typename Data>
class FixedSizeConQueue : public ConcurrentQueue<Data> {
public:
    explicit FixedSizeConQueue(size_t max_count) : ConcurrentQueue<Data>(),
                                                   max_count(max_count) {
        pthread_cond_init(&this->_full_cond, NULL);
    }

    virtual ~FixedSizeConQueue() {}

    virtual void push(const Data& data) {
        pthread_mutex_lock(&this->_mutex);
        while (this->_queue.size() >= this->max_count) {
            pthread_cond_wait&this->_full_cond, &this->_mutex);
        }

        this->_queue.push(data);

        pthread_cond_signal(this->_empty_cond);
        pthread_mutex_unlock(&this->_mutex);
    }

    virtual bool try_push(const Data& data) {
        bool is_success = false;
        pthread_mutex_lock(&this->_mutex);

        if (this->_queue.size() < this->max_count) {
            this->_queue.push(data);
            pthread_cond_signal(&this->_empty_cond);
            is_success = true;
        }

        pthread_mutex_unlock(&this->_mutex);
        return is_success;
    }

    virtual void pop(Data* data) {
        pthread_mutex_lock(&this->_mutex);
        while (this->_queue.empty()) {
            pthread_cond_wait(&this->_empty_cond, &this->_mutex);
        }

        *data = this->_queue.front();
        this->_queue.pop();

        pthread_cond_signal(&this->_full_cond);
        pthread_mutex_unlock(&this->_mutex);
    }

    virtual bool try_pop(Data* data) {
        bool is_success = false;
        pthread_mutex_lock(&this->_mutex);

        if (!this->_queue.empty()) {
            *data = this->_queue.front();
            this->_queue.pop();

            pthread_cond_signal(&this->_full_cond);
            is_success = true;
        }
        return is_success;
    }

    bool full() const {
        return this->_queue.size() >= this->max_count;
    }

    const size_t max_count = 0;

private:
    DISALLOW_COPY_AND_ASSIGN(FixedSizeConQueue);
    pthread_cond_t _full_cond;
};

#endif  // CONCURRENT_QUEUE_H