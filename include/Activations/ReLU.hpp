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
 * ReLU activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: ReLU.
 *
 * forward: output = input if input > 0, else 0
 * backward: output = 1*input if forward input was > 0, else 0
 */
class ReLU : public Module {
public:
  ReLU();
  ~ReLU() = default;

  /**
   * Forward pass of the ReLU activation function.
   *
   * @param[out] out input if input > 0, else 0
   * @param[in] x Values on which to apply ReLU
   */
  void forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) override;

  /**
   * Backward pass of the ReLU activation function.
   *
   * @param[out] din 1*input if forward input was > 0, else 0
   * @param[in] dout Values on which to apply backpropagation
   */
  void backward(Eigen::MatrixXf &din, const Eigen::MatrixXf &dout) override;

  /* Print description of ReLU activation class */
  void printDescription() override;

  /* Override set learning rate */
  void setLR(float lr) override {}

  /* Override getParametersCount */
  uint32_t getParametersCount() override { return 0; }

  std::string getName();

private:
  std::string _type = "Activation";
  std::string _name = "ReLU";
  Eigen::MatrixXf _forward_input;
};
}; // namespace Activations
}; // namespace DeepLearningFramework
