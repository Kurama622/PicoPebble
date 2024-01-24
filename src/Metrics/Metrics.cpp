/**
 * Metrics class implementation
 */

#include "Metrics.hpp"

using namespace DeepLearningFramework;

void Metrics::accuracy(float &accuracy, const Eigen::MatrixXf &labels,
                       const Eigen::MatrixXf &features) {
  accuracy = 0.f;
  for (int i = 0; i < features.rows(); i++) {
    if (features(i, 0) > features(i, 1) && labels(i, 0) > labels(i, 1))
      accuracy += 1.f;
    else if (features(i, 0) < features(i, 1) && labels(i, 0) < labels(i, 1))
      accuracy += 1.f;
  }
  accuracy /= features.rows();
}
