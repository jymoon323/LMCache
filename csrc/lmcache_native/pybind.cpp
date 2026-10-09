// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ttl_lock.h"
#include "bitmap.h"
#include "copy_pool.h"
#include "nt_memcpy.h"
#include "fold.h"
#include "periodic_event_notifier.h"
#include "utils.h"
#include "engine_kv_format.h"
#include "kv_transfer_plan_types.h"
#include "kv_transfer_types.h"

namespace py = pybind11;

using lmcache::lmcache_native::DevDaxCopyBatch;
using lmcache::lmcache_native::DevDaxCopyPool;
using lmcache::lmcache_native::DevDaxCopyRequest;
using lmcache::lmcache_native::DevDaxCopyStores;

using lmcache::lmcache_native::Bitmap;
using lmcache::lmcache_native::PeriodicEventNotifier;
using lmcache::lmcache_native::TTLLock;
using lmcache::utils::ParallelPatternMatcher;
using lmcache::utils::RangePatternMatcher;

PYBIND11_MODULE(lmcache_native, m) {
  m.doc() = "Native operations for LMCache (device-independent)";

  // Device-independent KV-format and transfer enums. These live in the
  // common (non-CUDA) native module so they are available regardless of the
  // GPU backend, and so they are the single source of truth shared with the
  // pure-Python fallback in ``lmcache.v1.platform.ops_types``.
  py::enum_<EngineKVFormat>(m, "EngineKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS)
      .value("NL_X_TWO_NB_NH_ONE_BS_HS",
             EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS)
      .value("NL_X_TWO_X_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS)
      .value("NL_X_NP_X_NB_BS_ONE_HS", EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS)
      .value("NL_X_NB_BS_NH_HS", EngineKVFormat::NL_X_NB_BS_NH_HS);

  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");

  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H);

  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems);

  py::class_<KernelGroupSpec>(m, "KernelGroupSpec")
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
                       int engine_kv_format, uintptr_t block_ids_base,
                       int64_t block_ids_capacity) {
             return KernelGroupSpec{
                 paged_buffer_ptrs,
                 std::move(lmcache_objects_ptrs),
                 shape_desc,
                 lmcache_chunk_size,
                 static_cast<EngineKVFormat>(engine_kv_format),
                 block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));

  m.def("is_kv_list", &is_kv_list, py::arg("format"),
        "Return whether the format stores KV as a list of per-token KV "
        "tensors (kv_size == 2).");
  m.def("is_layer_list", &is_layer_list, py::arg("format"),
        "Return whether the format stores one list entry per layer.");
  m.def("is_cross_layer", &is_cross_layer, py::arg("format"),
        "Return whether the format stacks KV from different layers into a "
        "single tensor.");
  m.def("is_mla", &is_mla, py::arg("format"),
        "Return whether the format is an MLA variant (single latent KV "
        "head).");
  m.def("is_kv_second_tuple", &is_kv_second_tuple, py::arg("format"),
        "Return whether each per-layer list entry is a (K, V) tuple of "
        "paged tensors.");

  m.def("fold", &lmcache::lmcache_native::fold, py::arg("found"),
        py::arg("num_chunks"), py::arg("num_ranks"), py::arg("group_windows"),
        "Fold per-(group, chunk, rank) presence into a servable-prefix-lengths "
        "bitmap (size num_chunks + 1); bit L set iff every object group can "
        "serve a length-L prefix.");
  m.def(
      "unfold", &lmcache::lmcache_native::unfold, py::arg("hit_length"),
      py::arg("num_chunks"), py::arg("num_ranks"), py::arg("group_windows"),
      "Expand a model-wide hit length into the per-group retain mask over the "
      "group x chunk x kv_rank layout.");
  m.def("fold_grouped", &lmcache::lmcache_native::fold_grouped, py::arg("rows"),
        py::arg("windows"),
        "Fold per-row presence bitmaps (rows[i] with window windows[i], all "
        "of size num_chunks) into a servable-prefix-lengths bitmap; bit j set "
        "iff every row can serve a length-(j + 1) prefix. Raises ValueError "
        "if rows and windows differ in length or the rows differ in size.");
  m.def("unfold_grouped", &lmcache::lmcache_native::unfold_grouped,
        py::arg("hit_length"), py::arg("num_chunks"), py::arg("windows"),
        "Expand a model-wide hit length into one retain bitmap of size "
        "num_chunks per window, parallel to windows.");

  m.def("devdax_memcpy_nt", &lmcache::lmcache_native::devdax_memcpy_nt,
        py::arg("dst"), py::arg("src"), py::arg("nbytes"),
        py::call_guard<py::gil_scoped_release>(),
        "Copy a Device-DAX gather segment to DRAM with AVX2 non-temporal "
        "stores, falling back to libc when AVX2 is unavailable or unsuitable.");
  m.def("devdax_memcpy_nt_backend",
        &lmcache::lmcache_native::devdax_memcpy_nt_backend,
        "Return avx2_nt when large gathers use AVX2 NT stores, else libc.");
  m.attr("DEVDAX_MEMCPY_NT_MIN_BYTES") =
      py::int_(lmcache::lmcache_native::kDevDaxMemcpyNtMinBytes);

  py::enum_<DevDaxCopyStores>(
      m, "DevDaxCopyStores",
      "Store instructions a DevDaxCopyPool submission uses: LIBC (memmove) "
      "or NON_TEMPORAL (devdax_memcpy_nt: AVX2 streaming stores for large "
      "segments, libc otherwise).")
      .value("LIBC", DevDaxCopyStores::kLibc)
      .value("NON_TEMPORAL", DevDaxCopyStores::kNonTemporal);

  py::class_<DevDaxCopyBatch, std::shared_ptr<DevDaxCopyBatch>>(
      m, "DevDaxCopyBatch",
      "Completion handle of one DevDaxCopyPool.submit(); settles once every "
      "copy of that submission has run.")
      .def("done", &DevDaxCopyBatch::done,
           "Whether every copy has finished (successfully or not).")
      .def(
          "wait",
          [](DevDaxCopyBatch& self) {
            // A settled batch (the common case for every handle but the
            // first one waited per batch) returns without a GIL round trip.
            if (!self.done()) {
              py::gil_scoped_release release;
              self.wait();
              return;
            }
            self.wait();
          },
          "Block until every copy has finished; re-raise the first copy "
          "error as RuntimeError.")
      .def_property_readonly("count", &DevDaxCopyBatch::count,
                             "Number of copies in the submission.")
      .def_property_readonly("nbytes", &DevDaxCopyBatch::nbytes,
                             "Bytes covered by the submission.");

  py::class_<DevDaxCopyPool>(
      m, "DevDaxCopyPool",
      "Fixed set of native copy threads shared by one process's staged "
      "retrieves. Copies run outside the Python interpreter, so the staging "
      "fan-out neither waits for nor contends with the GIL.")
      .def(py::init<std::size_t, std::vector<int>>(), py::arg("num_threads"),
           py::arg("cpus") = std::vector<int>{},
           "Start num_threads workers; when cpus is given (one per worker) "
           "each worker is pinned to its CPU, best effort.")
      .def(
          "submit",
          [](DevDaxCopyPool& self,
             const std::vector<std::tuple<std::uintptr_t, std::uintptr_t,
                                          std::size_t>>& copies,
             DevDaxCopyStores stores) {
            std::vector<DevDaxCopyRequest> requests;
            requests.reserve(copies.size());
            for (const auto& [dst, src, nbytes] : copies) {
              requests.push_back(DevDaxCopyRequest{dst, src, nbytes});
            }
            return self.submit(requests, stores);
          },
          py::arg("copies"), py::arg("stores") = DevDaxCopyStores::kLibc,
          "Queue (dst, src, nbytes) copies in order and return their "
          "DevDaxCopyBatch; the submission is atomic (all or none queued). "
          "stores selects libc memmove or the non-temporal devdax_memcpy_nt "
          "path. Raises RuntimeError once shutdown() has started.")
      .def("shutdown", &DevDaxCopyPool::shutdown,
           py::call_guard<py::gil_scoped_release>(),
           "Stop accepting submissions, finish every queued copy, and join "
           "the workers (safe to call concurrently and repeatedly).")
      .def_property_readonly("num_threads", &DevDaxCopyPool::num_threads)
      .def_property_readonly("pinned_threads", &DevDaxCopyPool::pinned_threads,
                             "Workers whose CPU pinning succeeded.")
      .def_property_readonly("busy_seconds", &DevDaxCopyPool::busy_seconds,
                             "Cumulative seconds the workers spent copying.")
      .def_property_readonly("completed_copies",
                             &DevDaxCopyPool::completed_copies)
      .def_property_readonly("queued", &DevDaxCopyPool::queued,
                             "Copies queued but not yet picked up.");

  py::class_<TTLLock>(m, "TTLLock")
      .def(py::init<uint32_t>(), py::arg("ttl_second") = 300,
           "Construct a TTLLock with the specified TTL duration in "
           "seconds. Default is 300 seconds.")
      .def("lock", &TTLLock::lock,
           "Increment the lock counter by 1 and update the TTL. "
           "If the previous TTL has expired, reset counter to 1.")
      .def("unlock", &TTLLock::unlock,
           "Decrement the lock counter by 1 (minimum 0).")
      .def("is_locked", &TTLLock::is_locked,
           "Check if the lock is held (counter > 0 and TTL not expired).")
      .def("reset", &TTLLock::reset,
           "Reset the lock to initial state (counter = 0, TTL expired).");

  py::class_<Bitmap>(m, "Bitmap")
      .def(py::init<size_t>(), py::arg("size"),
           "Construct a Bitmap with the specified size.")
      .def(py::init<size_t, size_t>(), py::arg("size"), py::arg("prefix_bits"),
           "Construct a Bitmap with the specified size and first N prefix "
           "bits set to 1.")
      .def("set", &Bitmap::set, py::arg("index"),
           "Set the bit at the specified index to 1.")
      .def("clear", &Bitmap::clear, py::arg("index"),
           "Clear the bit at the specified index to 0.")
      .def("test", &Bitmap::test, py::arg("index"),
           "Test the bit at the specified index.")
      .def("copy", &Bitmap::copy,
           "Return an independent copy with the same size and bits.")
      .def("popcount", &Bitmap::popcount, "Count the number of bits set to 1.")
      .def("size", &Bitmap::size, "Number of bits in the bitmap.")
      .def("__len__", &Bitmap::size, "Number of bits in the bitmap.")
      .def("count_leading_zeros", &Bitmap::clz,
           "Count the number of leading zeros.")
      .def("count_leading_ones", &Bitmap::clo,
           "Count the number of leading ones.")
      .def("highest_set_bit", &Bitmap::highest_set_bit,
           "Index of the highest set bit, or -1 if no bit is set (the return "
           "is signed so the empty bitmap is representable).")
      .def("__and__", &Bitmap::operator&, py::arg("other"),
           "Bitwise AND operation between two bitmaps.")
      .def("__or__", &Bitmap::operator|, py::arg("other"),
           "Bitwise OR operation between two bitmaps.")
      .def("__invert__", &Bitmap::operator~,
           "Bitwise NOT operation (flip all bits).")
      .def("get_indices_list", &Bitmap::get_indices,
           "Return a list of indices where the bit is set to 1.")
      .def("get_indices_set", &Bitmap::get_indices_set,
           "Return a set of indices where the bit is set to 1.")
      .def("batched_set", &Bitmap::batched_set, py::arg("indices"),
           "Set every bit in indices to 1 (positions >= size ignored).")
      .def("set_range", &Bitmap::set_range, py::arg("start"), py::arg("end"),
           "Set every bit in the half-open range [start, end) to 1 (end "
           "clamped to size). Fills whole bytes, so far cheaper than per-bit "
           "set for a contiguous span.")
      .def(
          "gather",
          [](const Bitmap& self, const py::sequence& items) {
            auto indices = self.get_indices();
            py::list result;
            for (auto idx : indices) {
              if (idx < static_cast<size_t>(py::len(items))) {
                result.append(items[idx]);
              }
            }
            return result;
          },
          py::arg("items"),
          "Return elements from items at indices where the bit is set to 1.")
      .def("__repr__", &Bitmap::to_string,
           "Convert the bitmap to a string representation.");

  py::class_<ParallelPatternMatcher>(m, "ParallelPatternMatcher")
      .def(py::init<const std::vector<int>&>(), py::arg("pattern"),
           "Construct a ParallelPatternMatcher with the specified pattern.")
      .def("match", &ParallelPatternMatcher::match, py::arg("data"),
           "Match the pattern in the given data and return a sorted list "
           "of positions where the pattern starts.");

  py::class_<RangePatternMatcher>(m, "RangePatternMatcher")
      .def(py::init<const std::vector<int>&, const std::vector<int>&>(),
           py::arg("start_pattern"), py::arg("end_pattern"),
           "Construct a RangePatternMatcher with start and end patterns. ")
      .def("match", &RangePatternMatcher::match, py::arg("data"),
           "Match ranges in the given data. Returns a list of (start_pos, "
           "end_pos) tuples where start_pos is the beginning of the start "
           "pattern and end_pos is the exclusive index after the end pattern. "
           "When multiple end patterns exist after a start pattern, matches "
           "the first one (minimal range).");

  py::class_<PeriodicEventNotifier>(m, "PeriodicEventNotifier")
      .def_static("create", &PeriodicEventNotifier::create,
                  py::call_guard<py::gil_scoped_release>(),
                  py::arg("interval_ms"), py::arg("use_eventfd"),
                  "Create the singleton PeriodicEventNotifier. "
                  "Idempotent -- second call is a no-op.")
      .def_static("get", &PeriodicEventNotifier::get,
                  py::return_value_policy::reference,
                  "Get the singleton instance, or None if not created.")
      .def_static("shutdown", &PeriodicEventNotifier::shutdown,
                  py::call_guard<py::gil_scoped_release>(),
                  "Shut down the singleton. Idempotent.")
      .def("register_fd", &PeriodicEventNotifier::register_fd, py::arg("fd"),
           "Register a file descriptor for periodic signaling.")
      .def("unregister_fd", &PeriodicEventNotifier::unregister_fd,
           py::arg("fd"), "Unregister a file descriptor.")
      .def("set_interval_ms", &PeriodicEventNotifier::set_interval_ms,
           py::arg("interval_ms"),
           "Change the notification interval in milliseconds.");
}
