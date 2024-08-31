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
