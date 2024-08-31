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
 * MSE loss class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Losses {
/**
 * Loss class: MSE.
 *
 * forward: output = input if input > 0, else 0
 * backward: output = 1*input if forward input was > 0, else 0
 */
class MSE {
public:
  MSE();
  ~MSE() = default;

  /**
   * Forward pass of the MSE loss function.
   *
   * @param[out] loss 1/N * SUM((yPred - y)^2), with N the number of samples
   * @param[in] y target values
   * @param[in] yPred values obtained by the neural network
   */
  void forward(float &loss, const Eigen::MatrixXf &y,
               const Eigen::MatrixXf &yPred);

  /**
   * Backward pass of the MSE loss function.
   *
   * @param[out] dloss 2*(yPred-y)/N, with N the number of samples
   * @param[in] y target values
   * @param[in] yPred values obtained by the neural network
   */
  void backward(Eigen::MatrixXf &dloss, const Eigen::MatrixXf &y,
                const Eigen::MatrixXf &yPred);

  /* Print description of MSE loss class */
  void printDescription();

  std::string getName();

private:
  std::string _type = "Loss";
  std::string _name = "MSE";
};
}; // namespace Losses
}; // namespace DeepLearningFramework
