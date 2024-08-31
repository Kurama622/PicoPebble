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
 * Softmax activation class definition
 */

#pragma once

#include "Module.hpp"

#include <iostream>

namespace DeepLearningFramework {
namespace Activations {
/**
 * Activation class: Softmax.
 *
 * forward: output = exp(IN_i)/exp(sum(IN)), input saved for backward pass
 * backward: output = [Softmax(forward_input) * (1 - Softmax(forward_input))] *
 * input
 */
class Softmax : public Module {
public:
  Softmax();
  ~Softmax() = default;

  /**
   * Forward pass of the Softmax activation function.
   *
   * @param[out] out exp(IN_i)/exp(sum(IN)), input saved for backward pass
   * @param[in] x Values on which to apply Softmax
   */
  void forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) override;

  /**
   * Backward pass of the Softmax activation function.
   *
   * @param[out] din [Softmax(forward_input) * (I - Softmax(forward_input))] *
   * input
   * @param[in] dout Values on which to apply backpropagation
   */
  void backward(Eigen::MatrixXf &din, const Eigen::MatrixXf &dout) override;

  /* Print description of Softmax activation class */
  void printDescription() override;

  /* Override set learning rate */
  void setLR(float lr) override {}

  /* Override getParametersCount */
  uint32_t getParametersCount() override { return 0; }

  std::string getName();

private:
  /**
   * Softmax equation implementation.
   *
   * @param[in] x Values on which to apply equation
   * @param[in] y exp(IN_i)/exp(sum(IN))
   */
  void equation(Eigen::MatrixXf &y, const Eigen::MatrixXf &x);

  std::string _type = "Activation";
  std::string _name = "Softmax";
  Eigen::MatrixXf _forward_input_with_softmax_applied;
};
}; // namespace Activations
}; // namespace DeepLearningFramework
