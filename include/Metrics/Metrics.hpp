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
