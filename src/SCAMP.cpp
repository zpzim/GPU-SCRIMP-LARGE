#include <chrono>
#include <cinttypes>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "SCAMP.h"
#include "common.h"
#include "cpu_stats.h"
#include "scamp_exception.h"
#include "tile.h"

using std::vector;

namespace SCAMP {

// This method computes all the work that must be done to perform the
// reqiested operation and sets the work order by populating the work
// queue.
void SCAMP_Operation::get_tiles() {
  size_t num_tile_rows = ceil((_info.full_ts_len_B - _info.mp_window + 1) /
                              static_cast<double>(_info.max_tile_height));
  size_t num_tile_cols = ceil((_info.full_ts_len_A - _info.mp_window + 1) /
                              static_cast<double>(_info.max_tile_width));
  if (!_info.silent_mode) {
    printf("num_tile_rows = %lu, cols = %lu\n", num_tile_rows, num_tile_cols);
  }

  if (_info.self_join) {
    for (int offset = 0; offset < num_tile_rows - 1; ++offset) {
      for (int diag = 0; diag < num_tile_cols - 1 - offset; ++diag) {
        _work_queue.push(std::make_pair(diag, diag + offset));
      }
    }

    for (int i = 0; i < num_tile_rows; ++i) {
      _work_queue.push(std::make_pair(i, num_tile_cols - 1));
    }
  } else {
    // Add upper diagonals one at a time except for edge tiles
    for (int diag = 0; diag < num_tile_cols - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_cols - 1 && offset < num_tile_rows - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset, diag + offset));
      }
    }

    // Add lower diagonals one at a time except for edge tiles
    for (int diag = 1; diag < num_tile_rows - 1; ++diag) {
      for (int offset = 0;
           offset + diag < num_tile_rows - 1 && offset < num_tile_cols - 1;
           ++offset) {
        _work_queue.push(std::make_pair(offset + diag, offset));
      }
    }

    // Add the corner edge tile
    _work_queue.push(std::make_pair(num_tile_rows - 1, num_tile_cols - 1));

    int x = 0;
    int y = 0;

    // Alternate between adding final row and final column edge tiles
    while (x < num_tile_cols - 1 && y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
      ++y;
    }

    // Add any remaining final row edge tiles
    while (x < num_tile_cols - 1) {
      _work_queue.push(std::make_pair(num_tile_rows - 1, x));
      ++x;
    }

    // Add any remaining final column edge tiles
    while (y < num_tile_rows - 1) {
      _work_queue.push(std::make_pair(y, num_tile_cols - 1));
      ++y;
    }
  }
  _total_tiles = _work_queue.size();
}

// This function is the point of entry for all worker threads
// Workers will initialize a memoty space to work with (Tile)
// and compute the partial matrix profile corresponding with
// the tile. Afterwards, the worker will merge its partial
// solution with the global matrix profile
void SCAMP_Operation::do_work(const std::vector<double> &timeseries_a,
                              const std::vector<double> &timeseries_b,
                              const OpInfo *info, const SCAMPArchitecture arch,
                              const int device_id) {
  // Init working memory and op/tile specific variables
  Tile tile(info, arch, device_id);
  if (!NeedsIntermittentReset(_info.profile_type)) {
    tile.InitProfile(_profile_a, _profile_b);
  }
  while (!_work_queue.empty()) {
    std::pair<int, int> t = _work_queue.pop();
    if (t.first == -1 && t.second == -1) {
      // Another thread grabbed our tile and now the queue is empty
      break;
    }
    // Get the position of the tile we will compute
    tile.set_tile_row(t.first * _info.max_tile_height);
    tile.set_tile_col(t.second * _info.max_tile_width);
    // Get the size of the tile we will compute
    tile.set_tile_width(std::min(_info.max_tile_ts_size,
                                 _info.full_ts_len_A - tile.get_tile_col()));
    tile.set_tile_height(std::min(_info.max_tile_ts_size,
                                  _info.full_ts_len_B - tile.get_tile_row()));
    if (!_info.silent_mode) {
      std::cout << "Starting tile with starting row of " << tile.get_tile_row()
                << " starting column of " << tile.get_tile_col()
                << " with height " << tile.get_tile_height() << " and width "
                << tile.get_tile_width() << std::endl;
    }
    // Copy the portion of the time series and stats
    //  we will be using from the global arrays.
    tile.InitTimeseries(timeseries_a, timeseries_b);
    tile.InitStats(_precompA, _precompB);
    tile.InitStats(_precomp);
    bool done = false;
    while (!done) {
      // Copy the portion of the best-so-far profile
      // we will be using.
      if (NeedsIntermittentReset(_info.profile_type)) {
        tile.InitProfile(_profile_a, _profile_b);
      }
      SCAMPError_t err;
      if (_info.self_join) {
        if (t.first == t.second) {
          // Partial tile on diagonal
          err = tile.execute(SELF_JOIN_UPPER_TRIANGULAR);
        } else {
          // Full Tile
          err = tile.execute(SELF_JOIN_FULL_TILE);
        }
      } else {
        // AB-join
        err = tile.execute(AB_FULL_JOIN_FULL_TILE);
      }
      if (err != SCAMP_NO_ERROR) {
        throw SCAMPException("ERROR " + getSCAMPErrorString(err) +
                             " executing tile");
      }
      // Merge join result
      if (NeedsIntermittentMerge(_info.profile_type)) {
        done = tile.MergeProfile(_profile_a, _profile_b);
      } else {
        done = true;
      }
    }
    // Update our counter with a lock
    std::unique_lock<std::mutex> lock(_counter_lock);
    _completed_tiles++;
  }
  if (!NeedsIntermittentMerge(_info.profile_type)) {
    tile.MergeProfile(_profile_a, _profile_b);
  }
}

// This method is the top level interface for a user to request
// that a join between timeseries_a and timeseries_b be computied
// using the configuration set up in SCAMP_Operation's constructor
SCAMPError_t SCAMP_Operation::do_join(const std::vector<double> &timeseries_a,
                                      const std::vector<double> &timeseries_b) {
  const int num_workers = _cpu_workers + _devices.size();
  if (!_info.silent_mode) {
    printf("Num workers = %d\n", num_workers);
  }

  std::vector<double> timeseries_a_clean, timeseries_b_clean;
  std::vector<bool> nanvals_a, nanvals_b;

  // Remove NaN/inf values and replace with 0, this allows us to tolerate these
  // values during the calculation. nanvals contains the subsequences which
  // contain non-finite values, we use these to force distance calculations
  // for these subsequences to result in NaN.
  convert_non_finite_to_zero(timeseries_a, _info.mp_window, &timeseries_a_clean,
                             &nanvals_a);
  convert_non_finite_to_zero(timeseries_b, _info.mp_window, &timeseries_b_clean,
                             &nanvals_b);

  // Compute statistics for entire problem
  compute_statistics_cpu(timeseries_a_clean, nanvals_a, &_precompA,
                         _info.mp_window);
  compute_statistics_cpu(timeseries_b_clean, nanvals_b, &_precompB,
                         _info.mp_window);

  _precomp = compute_combined_stats_cpu(timeseries_a_clean, timeseries_b_clean,
                                        _info.mp_window);

  // Populate Work Queue with tiles
  get_tiles();

  if (!_info.silent_mode) {
    std::cout << "Performing join with " << _work_queue.size() << " tiles."
              << std::endl;
  }
  std::vector<std::future<void>> futures(num_workers);

  if (!_info.silent_mode) {
    std::cout << "Main SCAMP thread spawning worker threads." << std::endl;
  }

  // Start CUDA Workers
  for (int i = 0; i < _devices.size(); ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this,
                            timeseries_a_clean, timeseries_b_clean, &_info,
                            CUDA_GPU_WORKER, _devices.at(i));
  }

  // Start CPU Workers
  for (int i = _devices.size(); i < num_workers; ++i) {
    futures[i] = std::async(std::launch::async, &SCAMP_Operation::do_work, this,
                            timeseries_a_clean, timeseries_b_clean, &_info,
                            CPU_WORKER, -1);
  }

  // wait for workers to be done
  for (auto &future : futures) {
    future.get();
  }

  return SCAMP_NO_ERROR;
}

void do_SCAMP(SCAMPArgs *args) {
  std::vector<int> devices;
  int num_threads = 0;
#ifdef _HAS_CUDA_
  int num_dev;
  cudaGetDeviceCount(&num_dev);
  for (int i = 0; i < num_dev; ++i) {
    devices.push_back(i);
  }
#endif
  if (devices.empty()) {
    num_threads = std::thread::hardware_concurrency();
  }
  do_SCAMP(args, devices, num_threads);
}

// Wrapper on SCAMP_Operation called by main(), this function constructs
// and executes a SCAMP_Operation given a set of user selected parameters.
void do_SCAMP(SCAMPArgs *args, const std::vector<int> &devices,
              int num_threads) {
  if (devices.empty() && num_threads <= 0) {
    throw SCAMPException("Error: no compute_resources provided");
  }

  if (args == nullptr) {
    throw SCAMPException("Error: Invalid arguments provided to SCAMP");
  }

  if (!args->silent_mode) {
    std::cout << "Validating SCAMP args." << std::endl;
  }
  args->validate();

  // Allocate and initialize memory
  OptionalArgs _opt_args(args->distance_threshold);

  if (!args->silent_mode) {
    std::cout << "Building SCAMP Operation from args" << std::endl;
  }

  // Construct operation
  SCAMP_Operation op(
      args->timeseries_a.size(), args->timeseries_b.size(), args->window,
      args->max_tile_size, devices, !args->has_b, args->precision_type,
      args->distributed_start_row, args->distributed_start_col, _opt_args,
      args->profile_type, &args->profile_a, &args->profile_b,
      args->keep_rows_separate, args->computing_rows, args->computing_columns,
      args->is_aligned, args->silent_mode, num_threads,
      args->max_matches_per_column, args->matrix_height, args->matrix_width);

  if (!args->silent_mode) {
    std::cout << "SCAMP Operation constructed" << std::endl;
  }
  // Execute op
  std::chrono::high_resolution_clock::time_point start =
      std::chrono::high_resolution_clock::now();
  if (args->has_b) {
    op.do_join(args->timeseries_a, args->timeseries_b);
  } else {
    op.do_join(args->timeseries_a, args->timeseries_a);
  }
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();
  if (!args->silent_mode) {
    printf(
        "Finished %d SCAMP tiles to generate  matrix profile in %lf "
        "seconds on %lu devices and %d threads\n",
        op.get_completed_tiles(),
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                .count() /
            static_cast<double>(1000000),
        devices.size(), num_threads);
  }
}

}  // namespace SCAMP
