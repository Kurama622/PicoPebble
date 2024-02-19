/**
 * Metrics class implementation
 */

#include "Metrics.hpp"
#include <iostream>

using namespace DeepLearningFramework;

void Metrics::accuracy(float &accuracy, const Eigen::MatrixXf &labels,
                       const Eigen::MatrixXf &features) {
  accuracy = 0.f;
  for(int i = 0; i < labels.rows(); i++) {
    int y_pred = -1;
    features.row(i).maxCoeff(&y_pred);
    if (y_pred == labels(i)) {
      accuracy+=1.f;
    }
  }
  accuracy /= features.rows();
}
