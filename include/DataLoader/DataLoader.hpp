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
 * DataLoader class definition
 */

#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace DeepLearningFramework {
/**
 * DataLoader class
 *
 * load: load dataset
 */
class DataLoader {
public:
  DataLoader() = delete;
  ~DataLoader() = delete;

  static void load(const std::string &path, Eigen::MatrixXf &X_train,
                   Eigen::MatrixXf &y_train, Eigen::MatrixXf &X_test,
                   Eigen::MatrixXf &y_test);

private:
  static Eigen::MatrixXf readMatrixFromFile(const std::string &filename);
  static std::vector<std::string> listFiles(const std::string &path);
  static void loadMatrix(const std::string &path,
                         Eigen::MatrixXf &concat_matrix);
};
}; // namespace DeepLearningFramework
