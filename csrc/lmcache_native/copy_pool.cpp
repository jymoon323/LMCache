// SPDX-License-Identifier: Apache-2.0

#include "copy_pool.h"

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
#endif

#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

#include "nt_memcpy.h"

namespace lmcache::lmcache_native {

// ----------------------------------------------------------- DevDaxCopyBatch

bool DevDaxCopyBatch::done() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return remaining_ == 0;
}

void DevDaxCopyBatch::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [&] { return remaining_ == 0; });
  if (!error_.empty()) {
    throw std::runtime_error(error_);
  }
}

void DevDaxCopyBatch::finish_one(const std::string& error) {
  // The decrement runs under the batch mutex so that a waiter which observes
  // remaining_ == 0 (and may then drop its last reference) cannot race with
  // this worker's notify: the worker still holds the mutex while notifying,
  // and the waiter cannot return from wait() before that mutex is released.
  std::lock_guard<std::mutex> lock(mutex_);
  if (!error.empty() && error_.empty()) {
    error_ = error;
  }
  if (remaining_ > 0 && --remaining_ == 0) {
    cv_.notify_all();
  }
}

// ------------------------------------------------------------ DevDaxCopyPool

DevDaxCopyPool::DevDaxCopyPool(std::size_t num_threads, std::vector<int> cpus)
    : num_threads_(num_threads) {
  if (num_threads == 0) {
    throw std::invalid_argument("DevDaxCopyPool needs at least one thread");
  }
  if (!cpus.empty() && cpus.size() != num_threads) {
    throw std::invalid_argument(
        "DevDaxCopyPool: cpus must be empty or hold one CPU per thread");
  }
  threads_.reserve(num_threads);
  try {
    for (std::size_t index = 0; index < num_threads; ++index) {
      const int cpu = cpus.empty() ? -1 : cpus[index];
      threads_.emplace_back(&DevDaxCopyPool::worker, this, cpu);
    }
  } catch (...) {
    // A worker that could not be created (thread limits) must not leave
    // the started ones joinable: their std::thread destructors would call
    // std::terminate while this exception unwinds the constructor.
    shutdown();
    throw;
  }
  // Let the caller read pinned_threads() right after construction.
  std::unique_lock<std::mutex> lock(mutex_);
  started_cv_.wait(lock, [&] { return started_ == threads_.size(); });
}

DevDaxCopyPool::~DevDaxCopyPool() {
  try {
    shutdown();
  } catch (...) {
    // A destructor must not throw; the join cannot fail in practice.
  }
}

std::shared_ptr<DevDaxCopyBatch> DevDaxCopyPool::submit(
    const std::vector<DevDaxCopyRequest>& copies, DevDaxCopyStores stores) {
  auto batch = std::make_shared<DevDaxCopyBatch>();
  batch->remaining_ = copies.size();
  batch->count_ = copies.size();
  for (const auto& copy : copies) {
    batch->nbytes_ += copy.nbytes;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_) {
      throw std::runtime_error("DevDaxCopyPool is shut down");
    }
    if (copies.empty()) {
      return batch;  // already done; the lifecycle contract still applies
    }
    const std::size_t queued_before = queue_.size();
    try {
      for (const auto& copy : copies) {
        queue_.push_back(Item{copy, stores, batch});
      }
    } catch (...) {
      // Keep the submission atomic: a copy the caller never got a handle
      // for must not run later against a bounce range it no longer owns.
      while (queue_.size() > queued_before) {
        queue_.pop_back();
      }
      throw;
    }
  }
  cv_.notify_all();
  return batch;
}

void DevDaxCopyPool::shutdown() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  cv_.notify_all();
  join_workers();
}

void DevDaxCopyPool::join_workers() {
  std::lock_guard<std::mutex> lock(shutdown_mutex_);
  if (joined_) {
    return;
  }
  for (auto& thread : threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  joined_ = true;
}

std::size_t DevDaxCopyPool::queued() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return queue_.size();
}

void DevDaxCopyPool::worker(int cpu) {
#if defined(__linux__)
  if (cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0) {
      pinned_.fetch_add(1);
    }
  }
#else
  // Pinning is Linux-only; elsewhere the worker stays unpinned and
  // pinned_threads() reports it (the caller logs a warning).
  (void)cpu;
#endif
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++started_;
  }
  started_cv_.notify_all();

  for (;;) {
    Item item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        // Only reached while stopping: queued copies always run first.
        return;
      }
      item = std::move(queue_.front());
      queue_.pop_front();
    }

    std::string error;
    const auto started = std::chrono::steady_clock::now();
    try {
      const auto& request = item.request;
      if (request.nbytes != 0) {
        if (item.stores == DevDaxCopyStores::kNonTemporal) {
          devdax_memcpy_nt(request.dst, request.src, request.nbytes);
        } else {
          if (request.dst == 0 || request.src == 0) {
            throw std::invalid_argument(
                "DevDaxCopyPool: null address for a non-empty copy");
          }
          std::memmove(reinterpret_cast<void*>(request.dst),
                       reinterpret_cast<const void*>(request.src),
                       request.nbytes);
        }
      }
    } catch (const std::exception& exc) {
      error = exc.what();
    } catch (...) {
      error = "DevDaxCopyPool: copy failed with a non-standard exception";
    }
    const auto finished = std::chrono::steady_clock::now();
    busy_ns_.fetch_add(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started)
            .count()));
    completed_.fetch_add(1);
    item.batch->finish_one(error);
  }
}

}  // namespace lmcache::lmcache_native
