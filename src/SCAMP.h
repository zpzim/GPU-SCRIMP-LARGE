#pragma once

#include <list>
#include <unordered_map>
#include <vector>
#include "common.h"

using std::list;
using std::pair;
using std::unordered_map;
using std::vector;

namespace SCAMP {

void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads);

void do_SCAMP(SCAMPArgs *args);

class SCAMP_Operation {
 private:
  // Precomputed statistics used by tiles
  PrecomputedInfo _precompA, _precompB;

  // Precomputed statistics computed from both input A and B.
  CombinedStats _precomp;

  // Result vectors
  Profile *_profile_a, *_profile_b;

  // Operation specific variables like maximum tile size (see common.h)
  const OpInfo _info;

  // Tile state variables
  // The order to compute the tiles in, set by get_tiles()
  ThreadSafeQueue _work_queue;

  // Lock for counter updates
  std::mutex _counter_lock;

  // The number of completed tiles
  int _completed_tiles;

  // The total number of tiles
  size_t _total_tiles;

  // Cuda devices to compute with
  std::vector<int> _devices;

  // CPU threads to compute with
  int _cpu_workers;

  void get_tiles();

  void do_work(const std::vector<double> &timeseries_a,
               const std::vector<double> &timeseries_b, const OpInfo *info,
               const SCAMPArchitecture arch, const int device_id);

 public:
  SCAMP_Operation(size_t Asize, size_t Bsize, size_t window_sz,
                  size_t max_tile_size, const vector<int> &dev, bool selfjoin,
                  SCAMPPrecisionType t, int64_t start_row, int64_t start_col,
                  OptionalArgs args_, SCAMPProfileType profile_type,
                  Profile *pA, Profile *pB, bool keep_rows, bool compute_rows,
                  bool compute_cols, bool is_aligned, bool silent_mode,
                  int num_threads, int64_t max_matches_per_col,
                  int64_t matrix_height, int64_t matrix_width)
      : _info(Asize, Bsize, window_sz, max_tile_size, selfjoin, t, start_row,
              start_col, args_, profile_type, keep_rows, compute_rows,
              compute_cols, is_aligned, silent_mode, dev.size() + num_threads,
              max_matches_per_col, matrix_height, matrix_width),
        _completed_tiles(0),
        _profile_a(pA),
        _profile_b(pB),
        _devices(dev),
        _cpu_workers(num_threads) {}

  SCAMPError_t do_join(const std::vector<double> &timeseries_a,
                       const std::vector<double> &timeseries_b);

  int get_completed_tiles() { return _completed_tiles; }
};

}  // namespace SCAMP
