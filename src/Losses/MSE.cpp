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
 * MSE loss class implementation
 */

#include "MSE.hpp"

#include <iostream>

using namespace DeepLearningFramework::Losses;

MSE::MSE() {}

void MSE::forward(float &loss, const Eigen::MatrixXf &y,
                  const Eigen::MatrixXf &y_pred) {
  loss = (y_pred - y).squaredNorm() / y.rows();
}

void MSE::backward(Eigen::MatrixXf &dloss, const Eigen::MatrixXf &y,
                   const Eigen::MatrixXf &y_pred) {
  dloss = 2.f * (y_pred - y) / y.rows();
}

void MSE::printDescription() { std::cout << "MSE loss" << std::endl; }

std::string MSE::getName() { return _name; }
