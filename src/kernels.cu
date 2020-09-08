#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <unordered_map>
#include "defines.h"
#include "kernel_common.h"
#include "kernel_gpu_utils.h"
#include "kernels.h"

namespace SCAMP {

/////////////////////////////////////////////////////////////////////////////////////
//     THESE HEADERS DEFINE COMPUTE STRATEGIES USED TO COMPUTE VARIOUS
//     PROFILE TYPES
///////////////////////////////////////////////////////////////////////////////////

#include "kernels_compute.h"
#include "kernels_smem.h"

// Computes the matrix profile given the sliding dot products for the first
// query and the precomputed data statisics
template <typename DATA_TYPE, typename VEC2_DATA_TYPE, typename VEC4_DATA_TYPE,
          typename ACCUM_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, typename DISTANCE_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, SCAMPProfileType PROFILE_TYPE, int blocks_per_sm,
          int tile_height, int BLOCKSZ>
__global__ void __launch_bounds__(BLOCKSZ, blocks_per_sm)
    do_tile(SCAMPKernelInputArgs args, PROFILE_OUTPUT_TYPE *profile_A,
            PROFILE_OUTPUT_TYPE *profile_B) {
  constexpr int tile_width = tile_height + BLOCKSZ * DIAGS_PER_THREAD;

  SCAMPThreadInfo<ACCUM_TYPE> thread_info;

  extern __shared__ char smem_raw[];

  // Wrap the shared memory in  a struct which contains handles shared memory
  // accesses
  SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> smem(
      smem_raw, COMPUTE_ROWS, COMPUTE_COLS, tile_width, tile_height,
      args.opt.num_extra_operands);

  // Find the starting diagonal of the distance matrix
  const unsigned int start_diag = args.exclusion_lower +
                                  (threadIdx.x * DIAGS_PER_THREAD) +
                                  blockIdx.x * (blockDim.x * DIAGS_PER_THREAD);

  // This is the index of the meta-diagonal that this thread block will work on
  const unsigned int meta_diagonal_idx = blockIdx.x;

  // The first diagonals constitiure a trivial match between the same
  // subsequence, we must exclude these from the calculation according to
  // args.exclusion_lower
  uint32_t tile_start_col =
      meta_diagonal_idx * (BLOCKSZ * DIAGS_PER_THREAD) + args.exclusion_lower;
  uint32_t tile_start_row = 0;

  // Initialize the column and row position of the current thread
  thread_info.global_col = tile_start_col + threadIdx.x * DIAGS_PER_THREAD;
  thread_info.global_row = 0;

  // num_diags is the number of diagonals in the distance matrix, less any
  // diagonals at the end we are not computing
  const unsigned int num_diags = args.n_x - args.exclusion_upper + 1;

  // Load the first dot product values
  if (thread_info.global_col < args.n_x) {
    thread_info.cov1 = args.cov[thread_info.global_col];
  }

  if (thread_info.global_col + 1 < args.n_x) {
    thread_info.cov2 = args.cov[thread_info.global_col + 1];
  }

  if (thread_info.global_col + 2 < args.n_x) {
    thread_info.cov3 = args.cov[thread_info.global_col + 2];
  }

  if (thread_info.global_col + 3 < args.n_x) {
    thread_info.cov4 = args.cov[thread_info.global_col + 3];
  }

  /////////////////////////////////////
  // Main loop
  /////////////////////////////////////
  // Each threadblock finds all the distances on a 'metadiagonal'
  // We use a tiled approach for each thread block
  // The tiles are horizontal slices of the diagonal, think of a parallelogram
  // cut from a diagonal slice of the distance matrix. Each thread starts on the
  // first row and works its way down-right towards right side of the distance
  // matrix
  while (tile_start_col < args.n_x && tile_start_row < args.n_y) {
    // Initialize the next tile's shared memory
    init_smem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_OUTPUT_TYPE, COMPUTE_ROWS,
              COMPUTE_COLS, tile_width, tile_height, BLOCKSZ>(
        args, smem, profile_A, profile_B, tile_start_col, tile_start_row);
    thread_info.local_col = threadIdx.x * DIAGS_PER_THREAD;
    thread_info.local_row = 0;

    // Start of new tile, sync so we don't have data races with shared memory
    // initializaton
    __syncthreads();

    // There are 2 pathways here, most of the time we take the fast path (top),
    // the last tile in every thread-block will take the slower path (bottom)
    if (tile_start_col + tile_width < args.n_x &&
        tile_start_row + tile_height < args.n_y &&
        start_diag + DIAGS_PER_THREAD <= num_diags) {
      // Fast Path
      while (thread_info.local_row < tile_height) {
        do_iteration_fast<DATA_TYPE, VEC2_DATA_TYPE, VEC4_DATA_TYPE, ACCUM_TYPE,
                          PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS,
                          COMPUTE_COLS, PROFILE_TYPE>(thread_info, smem,
                                                      args.opt);
      }

    } else if (start_diag < num_diags) {
      // Slow Path
      while (thread_info.global_col < args.n_x &&
             thread_info.global_row < args.n_y &&
             thread_info.local_row < tile_height) {
        do_row_edge<DATA_TYPE, PROFILE_DATA_TYPE, ACCUM_TYPE, DISTANCE_TYPE,
                    PROFILE_TYPE, COMPUTE_ROWS, COMPUTE_COLS>(
            thread_info, smem, args.n_x, start_diag, num_diags, args.opt);
        ++thread_info.global_col;
        ++thread_info.global_row;
        ++thread_info.local_col;
        ++thread_info.local_row;
      }
    }

    // After this sync, the caches will be updated with the best so far values
    // for this tile
    __syncthreads();

    // Write back our best-so-far computed for this tile to global memory
    write_back<DATA_TYPE, PROFILE_OUTPUT_TYPE, PROFILE_DATA_TYPE, COMPUTE_COLS,
               COMPUTE_ROWS, tile_width, tile_height, BLOCKSZ>(
        args, smem, tile_start_col, tile_start_row, args.n_x, args.n_y,
        profile_A, profile_B);

    // Update the tile position
    tile_start_col += tile_height;
    tile_start_row += tile_height;

    // Make sure our updates were committed before we pull in the next tile
    __threadfence_block();

    if (NeedsCheckIfDone(PROFILE_TYPE)) {
      // Copy the latest value of the profile length to shared memory
      if (threadIdx.x == 0) {
        *smem.profile_a_length = *args.profile_a_length;
        *smem.profile_b_length = *args.profile_b_length;
      }

      // Sync so that the write to shared memory is visible by all other threads
      __syncthreads();

      // If we have too many results, break this thread block out of the kernel
      // as more computation is pointless. We need to break the entire thread
      // block out at once otherwise this is undefined behavior.
      if (*smem.profile_a_length > args.max_matches_per_tile ||
          *smem.profile_b_length > args.max_matches_per_tile) {
        break;
      }
    }
  }
}

template <typename PROFILE_OUTPUT_TYPE, typename PROFILE_DATA_TYPE,
          typename DISTANCE_TYPE, SCAMPProfileType PROFILE_TYPE,
          int BLOCKSPERSM>
SCAMPError_t LaunchDoTile(SCAMPKernelInputArgs args,
                          PROFILE_OUTPUT_TYPE *profile_A,
                          PROFILE_OUTPUT_TYPE *profile_B,
                          SCAMPPrecisionType fp_type, bool computing_rows,
                          bool computing_cols, uint64_t blocksz,
                          uint64_t num_blocks, uint64_t smem, cudaStream_t s) {
  dim3 block(blocksz, 1, 1);
  dim3 grid(num_blocks, 1, 1);
  if (computing_rows && computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }

      default:
        return SCAMP_CUDA_ERROR;
    }
    return SCAMP_NO_ERROR;
  } else if (computing_cols) {
    constexpr bool COMPUTE_COLS = true;
    constexpr bool COMPUTE_ROWS = false;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      default:
        return SCAMP_CUDA_ERROR;
    }
  } else if (computing_rows) {
    constexpr bool COMPUTE_COLS = false;
    constexpr bool COMPUTE_ROWS = true;
    switch (fp_type) {
      case PRECISION_ULTRA:
      case PRECISION_DOUBLE: {
        do_tile<double, double2, double4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_DP, BLOCKSZ_DP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_MIXED: {
        do_tile<float, float2, float4, double, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      case PRECISION_SINGLE: {
        do_tile<float, float2, float4, float, PROFILE_OUTPUT_TYPE,
                PROFILE_DATA_TYPE, DISTANCE_TYPE, COMPUTE_ROWS, COMPUTE_COLS,
                PROFILE_TYPE, BLOCKSPERSM, TILE_HEIGHT_SP, BLOCKSZ_SP>
            <<<grid, block, smem, s>>>(args, profile_A, profile_B);
        break;
      }
      default:
        return SCAMP_CUDA_ERROR;
    }
  }
  gpuErrchk(cudaPeekAtLastError());
  return SCAMP_NO_ERROR;
}

SCAMPError_t compute_gpu_resources_and_launch(SCAMPKernelInputArgs args,
                                              Tile *t, void *profile_a,
                                              void *profile_b, bool do_rows,
                                              bool do_cols) {
  int exclusion_total = args.exclusion_lower + args.exclusion_upper;
  uint64_t blocksz = get_blocksz(t);
  uint64_t num_workers = ceil((args.n_x - exclusion_total) /
                              static_cast<double>(DIAGS_PER_THREAD));
  uint64_t num_blocks = ceil(num_workers / static_cast<double>(blocksz));
  uint64_t smem = get_smem(t->info(), blocksz);
  if (!t->info()->silent_mode) {
    std::cout << "Launching " << num_blocks << " thread blocks of size "
              << blocksz << " with a total of " << smem
              << " bytes of shared memory per block." << std::endl;
  }
  if (exclusion_total < args.n_x) {
    switch (t->info()->profile_type) {
      case PROFILE_TYPE_SUM_THRESH:
        return LaunchDoTile<double, double, double, PROFILE_TYPE_SUM_THRESH,
                            BLOCKSPERSM>(
            args, reinterpret_cast<double *>(profile_a),
            reinterpret_cast<double *>(profile_b), t->info()->fp_type, do_rows,
            do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_1NN_INDEX:
        return LaunchDoTile<uint64_t, uint64_t, float, PROFILE_TYPE_1NN_INDEX,
                            BLOCKSPERSM>(
            args, reinterpret_cast<uint64_t *>(profile_a),
            reinterpret_cast<uint64_t *>(profile_b), t->info()->fp_type,
            do_rows, do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_1NN:
        return LaunchDoTile<float, float, float, PROFILE_TYPE_1NN, BLOCKSPERSM>(
            args, reinterpret_cast<float *>(profile_a),
            reinterpret_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_APPROX_ALL_NEIGHBORS:
        return LaunchDoTile<SCAMPmatch, uint64_t, float,
                            PROFILE_TYPE_APPROX_ALL_NEIGHBORS, BLOCKSPERSM>(
            args, reinterpret_cast<SCAMPmatch *>(profile_a),
            reinterpret_cast<SCAMPmatch *>(profile_b), t->info()->fp_type,
            do_rows, do_cols, blocksz, num_blocks, smem, t->get_stream());
      case PROFILE_TYPE_MATRIX_SUMMARY:
        return LaunchDoTile<float, uint64_t, float, PROFILE_TYPE_MATRIX_SUMMARY,
                            BLOCKSPERSM>(
            args, reinterpret_cast<float *>(profile_a),
            reinterpret_cast<float *>(profile_b), t->info()->fp_type, do_rows,
            do_cols, blocksz, num_blocks, smem, t->get_stream());
      default:
        return SCAMP_FUNCTIONALITY_UNIMPLEMENTED;
    }
  }
  return SCAMP_NO_ERROR;
}

SCAMPError_t gpu_kernel_self_join_upper(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, false, false);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_self_join_lower(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, true, false);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

SCAMPError_t gpu_kernel_ab_join_upper(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, false, true);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_a(), t->profile_b(), t->info()->computing_rows,
      t->info()->computing_cols);
}

SCAMPError_t gpu_kernel_ab_join_lower(Tile *t) {
  SCAMPKernelInputArgs tile_args(t, true, true);
  return compute_gpu_resources_and_launch(
      tile_args, t, t->profile_b(), t->profile_a(), t->info()->computing_cols,
      t->info()->computing_rows);
}

void match_gpu_sort(SCAMPmatch *matches, int64_t len, cudaStream_t stream) {
  thrust::device_ptr<SCAMPmatch> ptr = thrust::device_pointer_cast(matches);
  thrust::sort(thrust::cuda::par.on(stream), ptr, ptr + len);
}

}  // namespace SCAMP
