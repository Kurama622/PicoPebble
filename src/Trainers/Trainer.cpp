/**
 * Trainer class implementation
 */

#include "Trainer.hpp"

using namespace DeepLearningFramework;

void Trainer::addAccuracy(std::vector<float> &accuracyHistory,
                            Sequential &model, const Eigen::MatrixXf &labels,
                            const Eigen::MatrixXf &features) {
  Eigen::MatrixXf tmpFeatures = features;
  model.forward(tmpFeatures);
  float accuracy = 0.f;
  Metrics::accuracy(accuracy, labels, tmpFeatures);
  accuracyHistory.push_back(accuracy);
}
