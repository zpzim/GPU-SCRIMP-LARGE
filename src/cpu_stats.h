#pragma once
#include <vector>
#include "common.h"

namespace SCAMP {

void convert_non_finite_to_zero(const std::vector<double> &T, const int m,
                                std::vector<double> *timeseries_clean,
                                std::vector<bool> *nanvals);

void compute_statistics_cpu(const std::vector<double> &T,
                            const std::vector<bool> &nanvalues,
                            PrecomputedInfo *info, size_t m);

CombinedStats compute_combined_stats_cpu(const std::vector<double> &A,
                                         const std::vector<double> &B,
                                         size_t m);

}  // namespace SCAMP
