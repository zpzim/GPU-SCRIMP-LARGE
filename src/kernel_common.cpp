#include "kernel_common.h"

namespace SCAMP {

SCAMPKernelInputArgs::SCAMPKernelInputArgs(Tile *t, bool transpose,
                                           bool ab_join) {
  cov = t->QT();
  dfa = transpose ? t->dfb() : t->dfa();
  dfb = transpose ? t->dfa() : t->dfb();
  dga = transpose ? t->dgb() : t->dga();
  dgb = transpose ? t->dga() : t->dgb();
  normsa = transpose ? t->normsb() : t->normsa();
  normsb = transpose ? t->normsa() : t->normsb();
  thresholds_a = transpose ? t->thresholds_B() : t->thresholds_A();
  thresholds_b = transpose ? t->thresholds_A() : t->thresholds_B();
  n_x = transpose ? t->get_tile_height() : t->get_tile_width();
  n_y = transpose ? t->get_tile_width() : t->get_tile_height();
  n_x = n_x - t->info()->mp_window + 1;
  n_y = n_y - t->info()->mp_window + 1;
  std::pair<int, int> exclusion =
      ab_join ? t->get_exclusion_for_ab_join(!transpose)
              : t->get_exclusion_for_self_join(!transpose);
  exclusion_lower = exclusion.first;
  exclusion_upper = exclusion.second;
  opt = t->info()->opt_args;
  profile_a_length =
      transpose ? t->get_mutable_b_dev_length() : t->get_mutable_a_dev_length();
  profile_b_length =
      transpose ? t->get_mutable_a_dev_length() : t->get_mutable_b_dev_length();
  max_matches_per_tile = t->info()->max_matches_per_tile;
  matrix_width = t->info()->matrix_width;
  matrix_height = t->info()->matrix_height;
  rows_per_cell = t->info()->rows_per_cell;
  cols_per_cell = t->info()->cols_per_cell;
  global_start_col = t->get_tile_col();
  global_start_row = t->get_tile_row();
}

void SCAMPKernelInputArgs::Print() {
  std::cout << "cov = " << cov << std::endl;
  std::cout << "dfa = " << dfa << std::endl;
  std::cout << "dfb = " << dfb << std::endl;
  std::cout << "dga = " << dga << std::endl;
  std::cout << "dgb = " << dgb << std::endl;
  std::cout << "normsa = " << normsa << std::endl;
  std::cout << "normsb = " << normsb << std::endl;
  std::cout << "max_matches_per_tile = " << max_matches_per_tile << std::endl;
  std::cout << "n_x = " << n_x << std::endl;
  std::cout << "n_y  = " << n_y << std::endl;
  std::cout << "exclusion_upper = " << exclusion_upper << std::endl;
  std::cout << "exclusion_lower = " << exclusion_lower << std::endl;
}

}  // namespace SCAMP
