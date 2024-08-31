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
 * Trainer class definition
 */

#pragma once

#include "Metrics.hpp"
#include "Sequential.hpp"

namespace DeepLearningFramework {
/**
 * Trainer class
 *
 * trainModel: train a model
 */
class Trainer {
public:
  Trainer() = delete;
  ~Trainer() = delete;

  /**
   * trainModel static method
   *
   * Train a model for n epoch on specified data
   *
   * @param[out] train_loss loss from epoch 0 to epochsCount on train set
   * @param[out] train_acc accuracy from epoch 0 to epochsCount on
   * train set
   * @param[out] test_loss loss from epoch 0 to epochsCount on test set
   * @param[out] test_acc accuracy from epoch 0 to epochsCount on
   * test set
   * @param[in/out] model to train
   * @param[in] epochsCount number of epochs
   * @param[in] y_train labels of the train set
   * @param[in] X_train features of the train set
   * @param[in] y_test labels of the test set
   * @param[in] X_test features of the test set
   * @param[in] batch_size batch size to use
   * @param[in] step display loss and metrics every N epochs
   * (default N=1)
   */
  template <uint32_t batch_size, uint32_t feature_dim>
  static void
  trainModel(std::vector<float> train_acc, std::vector<float> test_acc,
             Sequential &model, uint32_t epochs, const Eigen::MatrixXf &y_train,
             const Eigen::MatrixXf &X_train, const Eigen::MatrixXf &y_test,
             const Eigen::MatrixXf &X_test, uint32_t step);

private:
  /**
   * Calculate and add current accuracy to history
   *
   * Train a model for n epoch on specified data
   *
   * @param[out] accuracyHistory vector in which to add the accuracy
   * @param[in] model model to score wit haccuracy metric
   * @param[in] labels labels
   * @param[in] features features
   */
  static void addAccuracy(std::vector<float> &accuracyHistory,
                          Sequential &model, const Eigen::MatrixXf &labels,
                          const Eigen::MatrixXf &features);
};
}; // namespace DeepLearningFramework

#include "TrainerTemplateImpl.hpp"
