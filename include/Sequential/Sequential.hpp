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
 * Sequential model class definition
 */

#pragma once

#include "MSE.hpp"
#include "Module.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace DeepLearningFramework {
/**
 * Sequential class.
 *
 * Class used to store in sequence multiple modules to create a neural network
 *
 * forward: apply forward pass for each module in sequence
 * backward: calculate loss and apply backward pass for each layer in reverse
 * order.
 */
class Sequential {
public:
  Sequential(std::vector<Module *> &model, Losses::MSE loss);
  ~Sequential() {
    std::vector<Module *>::iterator it;
    for (it = _model.begin(); it != _model.end(); it++)
      delete (*it);
  }

  /**
   * Apply forward pass for each layer in sequence.
   *
   * @param[in/out] x data on which to apply the model (all layers in sequence).
   * Modified with neural network result
   */
  void forward(Eigen::MatrixXf &x, const std::string &mode = "train");

  /**
   * Calculate loss and apply backward pass for each layer in reverse order.
   *
   * @param[out] loss Loss value
   * @param[in] y target results.
   * @param[in] y_pred obtained results from the neural network.
   */
  void backward(float &loss, const Eigen::MatrixXf &y, Eigen::MatrixXf &y_pred);

  /* Print description of each module in sequence */
  void printDescription();

  /**
   * Set learning rate used to update weights for all modules
   *
   * @param[in] lr learning rate to use.
   */
  void setLR(float lr);

  /** Get the number of parameters of the model. */
  uint32_t getParametersCount();

  std::string getName();

private:
  // type, name, neural network
  std::string _type = "Module";
  std::string _name = "Sequential";
  std::vector<Module *> _model;
  Losses::MSE _loss;
  int _forward_flag;
  int _backward_flag;
};
}; // namespace DeepLearningFramework
