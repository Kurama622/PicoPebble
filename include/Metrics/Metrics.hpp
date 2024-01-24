/**
 * Metrics class definition
 */

#pragma once

#include <Eigen/Dense>

namespace DeepLearningFramework {
/**
 * Metrics class
 *
 * accuracy: count of good predictions / number of predictions
 */
class Metrics {
public:
  Metrics() = delete;
  ~Metrics() = delete;

  /**
   * accuracy static method
   *
   * accuracy: count of good predictions / number of predictions
   *
   * @param[out] accuracy accuracy in range [0.f, 1.f]
   * @param[in] labels one-hot encoded labels in format [N, 2]
   * @param[in] features prediction in format [N, 2]
   */
  static void accuracy(float &accuracy, const Eigen::MatrixXf &labels,
                       const Eigen::MatrixXf &features);
};
}; // namespace DeepLearningFramework
