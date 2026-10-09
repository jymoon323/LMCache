// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lmcache::lmcache_native {

// One raw-address copy queued on a DevDaxCopyPool.
struct DevDaxCopyRequest {
  std::uintptr_t dst;
  std::uintptr_t src;
  std::size_t nbytes;
};

// Store instruction family a submission uses: libc memmove, or the AVX2
// non-temporal path of devdax_memcpy_nt (falls back to libc itself when the
// CPU/build lacks AVX2 or the copy is small).
enum class DevDaxCopyStores { kLibc, kNonTemporal };

// Completion handle of one DevDaxCopyPool::submit(): it settles once every
// copy of that submission has run. The pool's workers never touch the Python
// interpreter, so a caller waiting here (with the GIL released) does not
// stall the copies behind Python threads, and the copies do not stall behind
// the caller.
class DevDaxCopyBatch {
 public:
  DevDaxCopyBatch() = default;
  DevDaxCopyBatch(const DevDaxCopyBatch&) = delete;
  DevDaxCopyBatch& operator=(const DevDaxCopyBatch&) = delete;

  // Whether every copy has finished (successfully or not).
  bool done() const;

  // Block until every copy has finished, then rethrow the first copy error
  // as std::runtime_error.
  void wait();

  // Number of copies and bytes in the submission.
  std::size_t count() const { return count_; }
  std::size_t nbytes() const { return nbytes_; }

 private:
  friend class DevDaxCopyPool;

  void finish_one(const std::string& error);

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::size_t remaining_ = 0;
  std::size_t count_ = 0;
  std::size_t nbytes_ = 0;
  std::string error_;
};

// Fixed set of copy threads shared by one process's staged retrieves.
//
// Copies run entirely outside the Python interpreter: a submission is one
// call, and completion is observed through DevDaxCopyBatch. This keeps the
// gather/scatter fan-out off the GIL, which a Python thread pool cannot do
// (each of its tasks re-acquires the GIL to start and to finish, so the copy
// workers convoy behind the request handlers' Python bursts).
class DevDaxCopyPool {
 public:
  // Start num_threads workers. When cpus is non-empty it must hold one CPU
  // per worker; worker i is then pinned to cpus[i] (best effort and
  // Linux-only -- see pinned_threads()). The constructor returns once every
  // worker is running;
  // if a worker cannot be created, the ones already started are joined and
  // the std::system_error propagates.
  DevDaxCopyPool(std::size_t num_threads, std::vector<int> cpus);
  ~DevDaxCopyPool();

  DevDaxCopyPool(const DevDaxCopyPool&) = delete;
  DevDaxCopyPool& operator=(const DevDaxCopyPool&) = delete;

  // Queue copies (in order) and return their completion handle. The
  // submission is atomic: either every copy is queued or none is (a failed
  // queue insertion rolls the earlier ones back before rethrowing). Throws
  // std::runtime_error once shutdown() has started.
  std::shared_ptr<DevDaxCopyBatch> submit(
      const std::vector<DevDaxCopyRequest>& copies, DevDaxCopyStores stores);

  // Stop accepting submissions, finish every queued copy, and join the
  // workers. Safe to call concurrently and repeatedly (later callers wait
  // for the first one's join); the destructor calls it. Never bounded: the
  // memory the copies touch is unmapped right after the owning
  // StagedRetrievePool closes, so a copy must not outlive the pool.
  void shutdown();

  std::size_t num_threads() const { return num_threads_; }
  // Workers whose CPU pinning succeeded (0 when no CPUs were given).
  std::size_t pinned_threads() const { return pinned_.load(); }
  // Cumulative time the workers spent inside copies, and copies finished.
  double busy_seconds() const { return busy_ns_.load() / 1e9; }
  std::uint64_t completed_copies() const { return completed_.load(); }
  // Copies queued but not yet picked up by a worker.
  std::size_t queued() const;

 private:
  struct Item {
    DevDaxCopyRequest request;
    DevDaxCopyStores stores;
    std::shared_ptr<DevDaxCopyBatch> batch;
  };

  void worker(int cpu);
  void join_workers();

  const std::size_t num_threads_;
  std::vector<std::thread> threads_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable started_cv_;
  std::deque<Item> queue_;
  std::size_t started_ = 0;
  bool stopping_ = false;
  // Serializes shutdown(): std::thread::join must run exactly once per
  // worker, and later callers must observe the join as complete.
  std::mutex shutdown_mutex_;
  bool joined_ = false;
  std::atomic<std::size_t> pinned_{0};
  std::atomic<std::uint64_t> busy_ns_{0};
  std::atomic<std::uint64_t> completed_{0};
};

}  // namespace lmcache::lmcache_native
