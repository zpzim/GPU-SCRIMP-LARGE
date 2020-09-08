#include "kernel_common.h"

namespace SCAMP {

template <typename T>
void SCAMPKernelInputArgs<T>::Print() {
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

template struct SCAMPKernelInputArgs<double>;

}  // namespace SCAMP
