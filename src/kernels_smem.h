#pragma once

/////////////////////////////////////////////////////
//
//
// STRATEGIES FOR INITIALIZING SHARED MEMORY
//
//
//////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
__device__ inline void init_smem_with_static_initializer(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
    uint32_t col_start, uint32_t row_start, PROFILE_DATA_TYPE initializer) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE, bool COMPUTE_ROWS,
          bool COMPUTE_COLS, int tile_width, int tile_height, int BLOCKSZ,
          SCAMPProfileType PROFILE_TYPE>
__device__ inline void init_smem_with_dynamic_initializer(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE> &smem,
    PROFILE_DATA_TYPE *initializer_col, PROFILE_DATA_TYPE *initializer_row,
    uint32_t col_start, uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      smem.local_mp_col[local_position] = initializer_col[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      smem.local_mp_row[local_position] = initializer_row[global_position];
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> &smem,
    PROFILE_OUTPUT_TYPE *profile_a, PROFILE_OUTPUT_TYPE *profile_b,
    uint32_t col_start, uint32_t row_start) {
  init_smem_with_dynamic_initializer<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS,
                                     COMPUTE_COLS, tile_width, tile_height,
                                     BLOCKSZ, PROFILE_TYPE_1NN>(
      args, smem, profile_a, profile_b, col_start, row_start);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> &smem,
    PROFILE_OUTPUT_TYPE *profile_a, PROFILE_OUTPUT_TYPE *profile_b,
    uint32_t col_start, uint32_t row_start) {
  init_smem_with_dynamic_initializer<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS,
                                     COMPUTE_COLS, tile_width, tile_height,
                                     BLOCKSZ, PROFILE_TYPE_1NN_INDEX>(
      args, smem, profile_a, profile_b, col_start, row_start);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> &smem,
    PROFILE_OUTPUT_TYPE *profile_a, PROFILE_OUTPUT_TYPE *profile_b,
    uint32_t col_start, uint32_t row_start) {
  init_smem_with_static_initializer<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS,
                                    COMPUTE_COLS, tile_width, tile_height,
                                    BLOCKSZ, PROFILE_TYPE_SUM_THRESH>(
      args, smem, col_start, row_start, 0.0);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> &smem,
    PROFILE_OUTPUT_TYPE *profile_a, PROFILE_OUTPUT_TYPE *profile_b,
    uint32_t col_start, uint32_t row_start) {
  mp_entry e;
  e.floats[0] = args.opt.threshold;
  e.ints[1] = 0;
  init_smem_with_static_initializer<DATA_TYPE, PROFILE_DATA_TYPE, COMPUTE_ROWS,
                                    COMPUTE_COLS, tile_width, tile_height,
                                    BLOCKSZ, PROFILE_TYPE_MATRIX_SUMMARY>(
      args, smem, col_start, row_start, e.ulong);
}

template <typename DATA_TYPE, typename PROFILE_DATA_TYPE,
          typename PROFILE_OUTPUT_TYPE, bool COMPUTE_ROWS, bool COMPUTE_COLS,
          int tile_width, int tile_height, int BLOCKSZ>
__device__ void init_smem(SCAMPKernelInputArgs &args,
                          SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE,
                                    PROFILE_TYPE_APPROX_ALL_NEIGHBORS> &smem,
                          PROFILE_OUTPUT_TYPE *profile_a,
                          PROFILE_OUTPUT_TYPE *profile_b, uint32_t col_start,
                          uint32_t row_start) {
  int global_position = col_start + threadIdx.x;
  int local_position = threadIdx.x;
  mp_entry initializer;
  initializer.ints[1] = 0;
  while (local_position < tile_width && global_position < args.n_x) {
    smem.dg_col[local_position] = args.dga[global_position];
    smem.df_col[local_position] = args.dfa[global_position];
    smem.inorm_col[local_position] = args.normsa[global_position];
    if (COMPUTE_COLS) {
      initializer.floats[0] = args.thresholds_a[global_position];
      smem.local_mp_col[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }

  global_position = row_start + threadIdx.x;
  local_position = threadIdx.x;
  while (local_position < tile_height && global_position < args.n_y) {
    smem.dg_row[local_position] = args.dgb[global_position];
    smem.df_row[local_position] = args.dfb[global_position];
    smem.inorm_row[local_position] = args.normsb[global_position];
    if (COMPUTE_ROWS) {
      initializer.floats[0] = args.thresholds_b[global_position];
      smem.local_mp_row[local_position] = initializer.ulong;
    }
    local_position += BLOCKSZ;
    global_position += BLOCKSZ;
  }
}

///////////////////////////////////////////////////////////////////
//
// STRATEGIES FOR WRITING BACK THE LOCAL MATRIX PROFILE TO MEMORY
//
///////////////////////////////////////////////////////////////////

template <typename DATA_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
__device__ void write_back(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_SUM_THRESH> &smem,
    uint32_t tile_start_x, uint32_t tile_start_y, uint32_t n_x, uint32_t n_y,
    PROFILE_OUTPUT_TYPE *profile_A, PROFILE_OUTPUT_TYPE *profile_B) {
  int global_position, local_position;
  if (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
          profile_A + global_position, smem.local_mp_col[local_position]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      do_atomicAdd<PROFILE_DATA_TYPE, ATOMIC_GLOBAL>(
          profile_B + global_position, smem.local_mp_row[local_position]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}

template <typename DATA_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
__device__ void write_back(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN> &smem,
    uint32_t tile_start_x, uint32_t tile_start_y, uint32_t n_x, uint32_t n_y,
    PROFILE_OUTPUT_TYPE *profile_A, PROFILE_OUTPUT_TYPE *profile_B) {
  int global_position, local_position;
  if (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      fAtomicMax<ATOMIC_GLOBAL>(profile_A + global_position,
                                smem.local_mp_col[local_position]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      fAtomicMax<ATOMIC_GLOBAL>(profile_B + global_position,
                                smem.local_mp_row[local_position]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}

template <typename DATA_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
__device__ void write_back(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_1NN_INDEX> &smem,
    uint32_t tile_start_x, uint32_t tile_start_y, uint32_t n_x, uint32_t n_y,
    PROFILE_OUTPUT_TYPE *profile_A, PROFILE_OUTPUT_TYPE *profile_B) {
  int global_position, local_position;
  if (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      mp_entry e;
      e.ulong = smem.local_mp_col[local_position];
      MPatomicMax<ATOMIC_GLOBAL>(profile_A + global_position, e.floats[0],
                                 e.ints[1]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      mp_entry e;
      e.ulong = smem.local_mp_row[local_position];
      MPatomicMax<ATOMIC_GLOBAL>(profile_B + global_position, e.floats[0],
                                 e.ints[1]);
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}

template <typename DATA_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
__device__ void write_back(
    SCAMPKernelInputArgs &args,
    SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE, PROFILE_TYPE_MATRIX_SUMMARY> &smem,
    uint32_t tile_start_x, uint32_t tile_start_y, uint32_t n_x, uint32_t n_y,
    PROFILE_OUTPUT_TYPE *profile_A, PROFILE_OUTPUT_TYPE *profile_B) {
  int global_position, local_position;
  if (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      mp_entry e;
      e.ulong = smem.local_mp_col[local_position];
      if (e.floats[0] > args.opt.threshold) {
        int col =
            (global_position + args.global_start_col) / args.cols_per_cell;
        int row = (e.ints[1] + args.global_start_row) / args.rows_per_cell;
        fAtomicMax<ATOMIC_GLOBAL>(profile_A + (row * args.matrix_width + col),
                                  e.floats[0]);
      }
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      mp_entry e;
      e.ulong = smem.local_mp_row[local_position];
      // In the matrix summary profile type, the only time we compute on rows in
      // in the transposed configuration, we can keep the col/row calculation
      // the same as above.
      if (e.floats[0] > args.opt.threshold) {
        int col =
            (global_position + args.global_start_col) / args.cols_per_cell;
        int row = (e.ints[1] + args.global_start_row) / args.rows_per_cell;
        fAtomicMax<ATOMIC_GLOBAL>(profile_B + (row * args.matrix_width + col),
                                  e.floats[0]);
      }
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}

template <typename DATA_TYPE, typename PROFILE_OUTPUT_TYPE,
          typename PROFILE_DATA_TYPE, bool COMPUTE_COLS, bool COMPUTE_ROWS,
          int TILE_WIDTH, int TILE_HEIGHT, int BLOCKSZ>
__device__ void write_back(SCAMPKernelInputArgs &args,
                           SCAMPSmem<DATA_TYPE, PROFILE_DATA_TYPE,
                                     PROFILE_TYPE_APPROX_ALL_NEIGHBORS> &smem,
                           uint32_t tile_start_x, uint32_t tile_start_y,
                           uint32_t n_x, uint32_t n_y,
                           PROFILE_OUTPUT_TYPE *profile_A,
                           PROFILE_OUTPUT_TYPE *profile_B) {
  int global_position, local_position;
  if (COMPUTE_COLS) {
    global_position = tile_start_x + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_WIDTH && global_position < n_x) {
      mp_entry e;
      e.ulong = smem.local_mp_col[local_position];
      if (e.floats[0] > args.thresholds_a[global_position]) {
        // Reserve space in output array
        unsigned long long int pos =
            do_atomicAdd<unsigned long long int, ATOMIC_GLOBAL>(
                args.profile_a_length, 1);
        // Write the match to the output
        if (pos < args.max_matches_per_tile) {
          profile_A[pos].corr = e.floats[0];
          profile_A[pos].row = e.ints[1];
          profile_A[pos].col = global_position;
        }
      }
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
  if (COMPUTE_ROWS) {
    global_position = tile_start_y + threadIdx.x;
    local_position = threadIdx.x;
    while (local_position < TILE_HEIGHT && global_position < n_y) {
      mp_entry e;
      e.ulong = smem.local_mp_row[local_position];
      if (e.floats[0] > args.thresholds_b[global_position]) {
        // Reserve space in output array
        unsigned long long int pos =
            do_atomicAdd<unsigned long long int, ATOMIC_GLOBAL>(
                args.profile_b_length, 1);
        // Write the match to the output
        if (pos < args.max_matches_per_tile) {
          profile_B[pos].corr = e.floats[0];
          profile_B[pos].row = e.ints[1];
          profile_B[pos].col = global_position;
        }
      }
      global_position += BLOCKSZ;
      local_position += BLOCKSZ;
    }
  }
}
