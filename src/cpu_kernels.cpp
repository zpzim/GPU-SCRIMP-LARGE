#include "cpu_kernels.h"
#include <array>
#include <vector>
#include "kernel_common.h"
namespace SCAMP {

// this is hard coded for now. It needs to be a power of 2
// It may be desirable to make this larger, since the compiler is generally
// unable to fold everything into register names with either int or long long
// based indexing.
constexpr int unrollWid{256};

// set here for now. Could be set by cmake depending on what is supported. AVX
// and AVX2 benefit from at least 32 particularly if masked movement
// instructions are generated for the reduction steps.
constexpr int simdByteLen{32};

// Outputs an 'initial' distance value based on the type of profile being
// computed
template <typename DISTANCE_TYPE, SCAMPProfileType type>
inline DISTANCE_TYPE init_dist() {
  switch (type) {
    case PROFILE_TYPE_KNN:
    case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
    case PROFILE_TYPE_1NN_INDEX:
    case PROFILE_TYPE_1NN:
      // Smallest value possible is -1 so set to -2
      return static_cast<DISTANCE_TYPE>(-2);
    case PROFILE_TYPE_SUM_THRESH:
    case PROFILE_TYPE_FREQUENCY_THRESH:
    default:
      // We must set to 0 so we get an accurate sum
      return static_cast<DISTANCE_TYPE>(0);
  }
}

template <SCAMPProfileType PROFILE_TYPE>
inline void update_mp(double *mp, double corr, int row, int col,
                      double thresh) {
  if (PROFILE_TYPE == PROFILE_TYPE_SUM_THRESH) {
    mp[col] = corr > thresh ? mp[col] + corr : mp[col];
  } else {
    ASSERT(false, "No Implementation provided for updating MP in CPU KERNEL");
  }
}

template <SCAMPProfileType PROFILE_TYPE>
inline void update_mp(mp_entry *mp, double corr, int row, int col,
                      double thresh) {
  if (PROFILE_TYPE == PROFILE_TYPE_1NN_INDEX) {
    if (corr > mp[col].floats[0]) {
      mp[col].floats[0] = corr;
      mp[col].ints[1] = row;
    }
  } else {
    ASSERT(false, "No Implementation provided for updating MP in CPU KERNEL");
  }
}

template <SCAMPProfileType PROFILE_TYPE>
inline void update_mp(float *mp, double corr, int row, int col, double thresh) {
  if (PROFILE_TYPE == PROFILE_TYPE_1NN) {
    mp[col] = mp[col] >= corr ? mp[col] : corr;
  } else {
    ASSERT(false, "No Implementation provided for updating MP in CPU KERNEL");
  }
}

template <typename DATA_TYPE, SCAMPProfileType type>
inline void reduce_row(std::array<DATA_TYPE, unrollWid> &corr,
                       std::array<int, unrollWid / 2> &corrIdx, double thresh) {
  switch (type) {
    case PROFILE_TYPE_1NN_INDEX: {
      for (int i = 0; i < unrollWid / 2; i++) {
        corrIdx[i] = corr[i] >= corr[i + unrollWid / 2] ? i : i + unrollWid / 2;
        corr[i] = corr[i] >= corr[i + unrollWid / 2] ? corr[i]
                                                     : corr[i + unrollWid / 2];
      }
      auto horizontal_reduction = [&corr, &corrIdx](int offset) {
        for (int i = 0; i < offset; i++) {
          corrIdx[i] =
              corr[i] >= corr[i + offset] ? corrIdx[i] : corrIdx[i + offset];
          corr[i] = corr[i] >= corr[i + offset] ? corr[i] : corr[i + offset];
        }
      };
      for (int i = unrollWid / 4; i > 0; i /= 2) {
        horizontal_reduction(i);
      }
      break;
    }
    case PROFILE_TYPE_1NN: {
      auto horizontal_reduction = [&corr](int offset) {
        for (int i = 0; i < offset; i++) {
          corr[i] = corr[i] >= corr[i + offset] ? corr[i] : corr[i + offset];
        }
      };
      for (int i = unrollWid / 2; i > 0; i /= 2) {
        horizontal_reduction(i);
      }
      break;
    }
    case PROFILE_TYPE_SUM_THRESH: {
      DATA_TYPE sum = 0;
      for (int i = 0; i < unrollWid; i++) {
        if (corr[i] > thresh) {
          sum += corr[i];
        }
      }
      corr[0] = sum;
      break;
    }
    default:
      break;
  }
}  // namespace SCAMP

template <typename DIST_TYPE, typename PROFILE_DATA_TYPE,
          SCAMPProfileType PROFILE_TYPE, bool computing_rows,
          bool computing_cols>
void do_tile(SCAMPKernelInputArgs &args,
             PROFILE_DATA_TYPE *__restrict profile_A,
             PROFILE_DATA_TYPE *__restrict profile_B) {
  DIST_TYPE initializer = init_dist<DIST_TYPE, PROFILE_TYPE>();
  int num_diags = args.n_x - args.exclusion_upper + 1;
  for (int tile_diag = args.exclusion_lower; tile_diag < num_diags;
       tile_diag += unrollWid) {
    // Determine the maximum number of iterations for this tile (includes slow
    // case)
    int rowIters = std::min(args.n_x - tile_diag, args.n_y);

    // Determine how many optimized iterations we can do before the slow case
    int fullRowIters;
    if (tile_diag + unrollWid >= num_diags) {
      fullRowIters = 0;
    } else {
      fullRowIters =
          std::max(0, std::min(args.n_x - tile_diag - unrollWid + 1, args.n_y));
    }

    // Fast, Unrolled, Autovectorized Case
    for (int row = 0; row < fullRowIters; row++) {
      alignas(simdByteLen) std::array<DIST_TYPE, unrollWid> corr;
      for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
        int curr_diag = tile_diag + local_diag;
        int col = curr_diag + row;
        DIST_TYPE correlation =
            args.cov[curr_diag] * args.normsa[col] * args.normsb[row];
        corr[local_diag] =
            std::isfinite(correlation) ? correlation : initializer;
      }
      if (computing_cols) {
        for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
          int curr_diag = tile_diag + local_diag;
          int col = curr_diag + row;
          update_mp<PROFILE_TYPE>(profile_A, corr[local_diag], row, col,
                                  args.opt.threshold);
        }
      }
      if (computing_rows) {
        std::array<int, unrollWid / 2> corrIdx;
        reduce_row<DIST_TYPE, PROFILE_TYPE>(corr, corrIdx, args.opt.threshold);
        update_mp<PROFILE_TYPE>(profile_B, corr[0],
                                corrIdx[0] + tile_diag + row, row,
                                args.opt.threshold);
      }
      for (int local_diag = 0; local_diag < unrollWid; local_diag++) {
        int curr_diag = tile_diag + local_diag;
        int col = curr_diag + row;
        args.cov[curr_diag] += args.dfa[col] * args.dgb[row];
        args.cov[curr_diag] += args.dfb[row] * args.dga[col];
      }
    }

    // Slow Case
    for (int row = fullRowIters; row < rowIters; row++) {
      int diagmax = std::min(
          std::min(args.n_x - args.exclusion_upper + 1, args.n_x - row),
          tile_diag + unrollWid);
      for (int diag = tile_diag; diag < diagmax; diag++) {
        int col = diag + row;
        DIST_TYPE corr = args.cov[diag] * args.normsa[col] * args.normsb[row];
        corr = std::isfinite(corr) ? corr : initializer;
        if (computing_cols) {
          update_mp<PROFILE_TYPE>(profile_A, corr, row, col,
                                  args.opt.threshold);
        }
        if (computing_rows) {
          update_mp<PROFILE_TYPE>(profile_B, corr, col, row,
                                  args.opt.threshold);
        }
      }
      for (int diag = tile_diag; diag < diagmax; diag++) {
        int col = diag + row;
        args.cov[diag] += args.dfa[col] * args.dgb[row];
        args.cov[diag] += args.dga[col] * args.dfb[row];
      }
    }
  }
}

template <typename DIST_TYPE, typename PROFILE_OUTPUT_TYPE,
          SCAMPProfileType PROFILE_TYPE>
SCAMPError_t LaunchDoTile(SCAMPKernelInputArgs &args,
                          PROFILE_OUTPUT_TYPE *profile_A,
                          PROFILE_OUTPUT_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          bool computing_cols) {
  if (computing_rows && computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  } else if (computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = false;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  } else if (computing_rows) {
    constexpr bool COMPUTE_COLS = false;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE:
        do_tile<DIST_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_TYPE, COMPUTE_ROWS,
                COMPUTE_COLS>(args, profile_A, profile_B);
        break;
      case PRECISION_MIXED:
      case PRECISION_SINGLE:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t compute_cpu_resources_and_launch(SCAMPKernelInputArgs &args,
                                              Tile *t, void *profile_a,
                                              void *profile_b, bool do_rows,
                                              bool do_cols) {
  int exclusion_total = args.exclusion_lower + args.exclusion_upper;
  if (exclusion_total < args.n_x) {
    switch (t->info()->profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double, PROFILE_TYPE_SUM_THRESH>(
            args, reinterpret_cast<double *>(profile_a),
            reinterpret_cast<double *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<float, mp_entry, PROFILE_TYPE_1NN_INDEX>(
            args, reinterpret_cast<mp_entry *>(profile_a),
            reinterpret_cast<mp_entry *>(profile_b), t->info()->fp_type,
            do_rows, do_cols);
      case PROFILE_TYPE_1NN:
        return LaunchDoTile<float, float, PROFILE_TYPE_1NN>(
            args, reinterpret_cast<float *>(profile_a),
            reinterpret_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols);
      case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
      case PROFILE_TYPE_MATRIX_SUMMARY:
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t cpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, false, false);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, true, false);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

SCAMPError_t cpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, false, true);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t cpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, true, true);
  return compute_cpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

};  // namespace SCAMP
