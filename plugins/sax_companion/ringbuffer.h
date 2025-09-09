#pragma once
#include <vector>
#include <cstddef>

template <typename T>
class RingBuffer {
public:
    explicit RingBuffer(std::size_t size) : buffer(size) {}

    bool push(const T& item) {
        auto next = (head + 1) % buffer.size();
        if (next == tail)
            return false;
        buffer[head] = item;
        head = next;
        return true;
    }

    bool pop(T& item) {
        if (tail == head)
            return false;
        item = buffer[tail];
        tail = (tail + 1) % buffer.size();
        return true;
    }

private:
    std::vector<T> buffer;
    std::size_t head{0};
    std::size_t tail{0};
};
