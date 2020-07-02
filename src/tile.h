#pragma once
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include "common.h"
#include "qt_helper.h"

namespace SCAMP {

class Tile {
 private:
  // Per worker input vectors
  std::unique_ptr<double, std::function<void(double *)>> _T_A_dev, _T_B_dev,
      _QT_dev, _means_A, _means_B, _norms_A, _norms_B, _df_A, _df_B, _dg_A,
      _dg_B, _scratchpad;

  std::unique_ptr<float, std::function<void(float *)>> _thresholds_A,
      _thresholds_B;
  // Per worker output vectors (device)
  DeviceProfile _profile_a_tile_dev, _profile_b_tile_dev;

  // Length of the output vector on the device
  // Set by the kernel when the profile can have a variable length
  unsigned long long int *_profile_a_dev_length;
  unsigned long long int *_profile_b_dev_length;

  // Vector of matches used for ALL_NEIGHBORS joins
  std::vector<SCAMPmatch> _matches_local;

  // Per worker output vectors (host)
  Profile _profile_a_tile, _profile_b_tile;

  // Helper for FFT computation
  std::shared_ptr<qt_compute_helper> _scratch;

  // Tile-specific variables describing the current tile
  size_t _current_tile_width;
  size_t _current_tile_height;
  size_t _current_tile_col;
  size_t _current_tile_row;

  const OpInfo *_info;
  ExecInfo _exec_info;

  // Private member functions which perform joins
  SCAMPError_t do_self_join_full();
  SCAMPError_t do_self_join_half();
  SCAMPError_t do_ab_join_full();

  void init();
  void init_cuda();
  void init_cpu();
  void free_cuda();
  void free_cpu();
  void Memset(void *destination, char value, size_t bytes);
  void Memcopy(void *destination, const void *source, size_t bytes,
               bool from_tile);
  Profile AllocProfile(SCAMPProfileType t, uint64_t size);
  void CopyProfileToHost(Profile *destination_profile,
                         const DeviceProfile *device_tile_profile,
                         uint64_t length);
  void SortMatches(SCAMPmatch *matches, uint64_t len);

  // Gets the profile length computed by the kernel when there can be a variable
  // length
  std::pair<int64_t, int64_t> get_profile_dims_from_device();

 public:
  Tile(const OpInfo *info);
  Tile(const OpInfo *info, SCAMPArchitecture arch, int cuda_id);
  ~Tile();
#ifdef _HAS_CUDA_
  cudaStream_t get_stream() { return _exec_info.stream; }
  cudaDeviceProp get_dev_props() { return _exec_info.dev_props; }
#endif

  int get_cuda_id() const {
#ifdef _HAS_CUDA_
    return _exec_info.cuda_id;
#else
    return -1;
#endif
  }
  SCAMPArchitecture get_arch() const { return _exec_info.arch; }
  size_t get_tile_width() const { return _current_tile_width; }
  size_t get_tile_height() const { return _current_tile_height; }
  size_t get_tile_row() const { return _current_tile_row; }
  size_t get_tile_col() const { return _current_tile_col; }
  const OpInfo *info() const { return _info; }
  void *profile_a() { return _profile_a_tile_dev.at(_info->profile_type); };
  void *profile_b() { return _profile_b_tile_dev.at(_info->profile_type); };
  double *QT() { return _QT_dev.get(); }
  const double *dfa() const { return _df_A.get(); }
  const double *dfb() const { return _df_B.get(); }
  const double *dga() const { return _dg_A.get(); }
  const double *dgb() const { return _dg_B.get(); }
  const double *normsa() const { return _norms_A.get(); }
  const double *normsb() const { return _norms_B.get(); }
  const float *thresholds_A() const { return _thresholds_A.get(); }
  const float *thresholds_B() const { return _thresholds_B.get(); }
  unsigned long long int *get_mutable_a_dev_length() {
    return _profile_a_dev_length;
  }
  unsigned long long int *get_mutable_b_dev_length() {
    return _profile_b_dev_length;
  }

  std::pair<int, int> get_exclusion_for_ab_join(bool upper_tile);
  std::pair<int, int> get_exclusion_for_self_join(bool upper_tile);
  void set_tile_col(size_t col) { _current_tile_col = col; }
  void set_tile_row(size_t row) { _current_tile_row = row; }
  void set_tile_height(size_t height) { _current_tile_height = height; }
  void set_tile_width(size_t width) { _current_tile_width = width; }
  bool MergeProfile(Profile *profile_a, Profile *profile_b);
  void Sync();

  // Initializes the precomputed statistics required by the current tile
  void InitStats(const PrecomputedInfo &a, const PrecomputedInfo &b);

  void InitStats(const CombinedStats &ab);

  // Initializes the ouptut vector with the current best-so-far profile
  SCAMPError_t InitProfile(Profile *profile_a, Profile *profile_b);

  // Initializes the time series for the current tile
  void InitTimeseries(const std::vector<double> &Ta_h,
                      const std::vector<double> &Tb_h);

  // Executes a pre-initialized tile
  SCAMPError_t execute(SCAMPTileType t);
};

}  // namespace SCAMP
