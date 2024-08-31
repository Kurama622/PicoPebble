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
 * Interface class for all the modules
 */

#pragma once

#include <Eigen/Dense>

namespace DeepLearningFramework {
class Module {
public:
  virtual ~Module() = default;

  virtual void forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) = 0;

  virtual void backward(Eigen::MatrixXf &ddout,
                        const Eigen::MatrixXf &dout) = 0;

  virtual void printDescription() = 0;

  virtual void setLR(float lr) = 0;

  virtual uint32_t getParametersCount() = 0;
  virtual std::string getName() = 0;
};
}; // namespace DeepLearningFramework
