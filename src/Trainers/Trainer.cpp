///////////////////////////////////////////////////////////////////////////
//
// PicoPebble - A lightweight distributed machine learning training framework for beginners
//
///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2024. All rights reserved.
//
// Licensed under the MIT License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://opensource.org/licenses/MIT
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
///////////////////////////////////////////////////////////////////////////
/**
 * Trainer class implementation
 */

#include "Trainer.hpp"

using namespace DeepLearningFramework;

void Trainer::addAccuracy(std::vector<float> &accuracyHistory,
                            Sequential &model, const Eigen::MatrixXf &labels,
                            const Eigen::MatrixXf &features) {
  Eigen::MatrixXf tmpFeatures = features;
  model.forward(tmpFeatures, "predict");
  float accuracy = 0.f;
  Metrics::accuracy(accuracy, labels, tmpFeatures);
  accuracyHistory.push_back(accuracy);
}
