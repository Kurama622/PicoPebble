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
 * ReLU layer class implementation
 */

#include "ReLU.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

ReLU::ReLU() {}

void ReLU::forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) {
  _forward_input = x;
  out = (x.array() < 0.f).select(0.f, x);
}

void ReLU::backward(Eigen::MatrixXf &din, const Eigen::MatrixXf &dout) {
  din = (_forward_input.array() < 0.f).select(0.f, dout);
}

void ReLU::printDescription() { std::cout << "ReLU activation" << std::endl; }

std::string ReLU::getName() { return _name; }
