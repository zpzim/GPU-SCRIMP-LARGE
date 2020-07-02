#include "cpu_stats.h"
#include <cmath>
#include <vector>
namespace SCAMP {

// Converts any NaN or inf values in the input to 0, returns the cleaned
// timeseries in timeseries_clean, and returns the subsequences which contained
// NaN in nanvals
void convert_non_finite_to_zero(const std::vector<double> &T, const int m,
                                std::vector<double> *timeseries_clean,
                                std::vector<bool> *nanvals) {
  timeseries_clean->resize(T.size());
  nanvals->resize(T.size() - m + 1);
  size_t steps_since_last_nan = m;
  for (int i = 0; i < T.size(); ++i) {
    if (std::isfinite(T[i])) {
      timeseries_clean->at(i) = T[i];
    } else {
      steps_since_last_nan = 0;
      timeseries_clean->at(i) = 0;
    }
    if (i >= m - 1) {
      nanvals->at(i - m + 1) = steps_since_last_nan < m;
    }
    steps_since_last_nan++;
  }
}

// moving mean over sequence a with window length w
// based on Ogita et. al, Accurate Sum and Dot Product

// A major source of rounding error is accumulated error in the mean values, so
// we use this to compensate. While the error bound is still a function of the
// conditioning of a very long dot product, we have observed a reduction of 3 -
// 4 digits lost to numerical roundoff when compared to older solutions.
std::vector<double> moving_mean(std::vector<double> a, int w) {
  std::vector<double> result(a.size() - w + 1, 0);
  double p = a[0];
  double s = 0;
  for (int i = 1; i < w; ++i) {
    double x = p + a[i];
    double z = x - p;
    s += (p - (x - z)) + (a[i] - z);
    p = x;
  }
  result[0] = (p + s) / w;

  for (int i = w; i < a.size(); ++i) {
    double x = p - a[i - w];
    double z = x - p;
    s += (p - (x - z)) - (a[i - w] + z);
    p = x;

    x = p + a[i];
    z = x - p;
    s += (p - (x - z)) + (a[i] - z);
    p = x;
    result[i - w + 1] = (p + s) / w;
  }
  return std::move(result);
}

// Computes all required statistics for SCAMP, populating info with these values
void compute_statistics_cpu(const std::vector<double> &T,
                            const std::vector<bool> &nanvalues,
                            PrecomputedInfo *info, size_t m) {
  std::vector<double> prefix_sum(T.size());
  int n = T.size() - m + 1;
  std::vector<double> norms(n), df(n), dg(n);

  std::vector<double> means = moving_mean(T, m);

  for (int subseq = 0; subseq < n; ++subseq) {
    if (nanvalues[subseq]) {
      norms[subseq] = std::nan("NaN");
    } else {
      double sum = 0;
      for (int i = 0; i < m; ++i) {
        double val = T[subseq + i] - means[subseq];
        sum += val * val;
      }
      if (sum == 0) {
        norms[subseq] = std::nan("NaN");
      } else {
        norms[subseq] = static_cast<double>(1.0) / std::sqrt(sum);
      }
    }
  }

  for (int i = 0; i < n - 1; ++i) {
    df[i] = (T[i + m] - T[i]) / static_cast<double>(2);
    dg[i] = (T[i + m] - means[i + 1]) + (T[i] - means[i]);
  }

  info->set(means, norms, df, dg);
}

CombinedStats compute_combined_stats_cpu(const std::vector<double> &A,
                                         const std::vector<double> &B,
                                         size_t m) {
  CombinedStats result;

  int na = A.size() - m + 1;
  int nb = B.size() - m + 1;
  int na2 = A.size() - (m - 1) + 1;
  int nb2 = B.size() - (m - 1) + 1;
  std::vector<double> dc_fwd(na2), dc_bkwd(na2), dr_fwd(nb2), dr_bkwd(nb2);

  std::vector<double> means_A = moving_mean(A, m);
  std::vector<double> means_B = moving_mean(B, m - 1);

  for (int i = 0; i < na; ++i) {
    dc_bkwd[i] = -1 * (A[i] - means_A[i]);
    dc_fwd[i] = (A[i + m] - means_A[i + 1]);
  }

  for (int i = 0; i < nb2; ++i) {
    dr_bkwd[i] = B[i] - means_B[i + 1];
    dr_fwd[i] = B[i + m] - means_B[i + 1];
  }

  result.dc_bkwd = std::move(dc_bkwd);
  result.dr_bkwd = std::move(dr_bkwd);
  result.dc_fwd = std::move(dc_fwd);
  result.dr_fwd = std::move(dr_fwd);

  return std::move(result);
}

}  // namespace SCAMP
