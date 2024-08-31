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
#include "DataLoader.hpp"
#include "Common.hpp"
#include "GlobalState.hpp"
#include "mpi/MpiController.hpp"
#include <dirent.h>
#include <fstream>
#include <iostream>

using namespace DeepLearningFramework;

void DataLoader::load(const std::string &path, Eigen::MatrixXf &X_train,
                      Eigen::MatrixXf &y_train, Eigen::MatrixXf &X_test,
                      Eigen::MatrixXf &y_test) {
  std::string X_train_path = path + "train_features/";
  std::string y_train_path = path + "train_labels/";
  std::string X_test_path = path + "test_features/";
  std::string y_test_path = path + "test_labels/";
  loadMatrix(X_train_path, X_train);
  loadMatrix(y_train_path, y_train);
  loadMatrix(X_test_path, X_test);
  loadMatrix(y_test_path, y_test);
}

void DataLoader::loadMatrix(const std::string &path,
                            Eigen::MatrixXf &concat_matrix) {
  std::vector<std::string> part_files = listFiles(path);

  for (int i = 0; i < part_files.size(); i++) {
    Eigen::MatrixXf matrix = readMatrixFromFile(part_files[i]);
    if (concat_matrix.rows() == 0) {
      concat_matrix.resize(matrix.rows() * part_files.size(), matrix.cols());
    }
    concat_matrix.block(matrix.rows() * i, 0, matrix.rows(), matrix.cols()) =
        matrix;
  }
}

Eigen::MatrixXf DataLoader::readMatrixFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << filename << std::endl;
    return Eigen::MatrixXf();
  }

  std::string line;
  std::vector<std::vector<float>> data;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream line_stream(line);
    float value;
    while (line_stream >> value) {
      row.push_back(value);
      if (line_stream.peek() == ',') {
        line_stream.ignore();
      }
    }
    data.push_back(row);
  }

  Eigen::MatrixXf matrix(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].size(); ++j) {
      matrix(i, j) = data[i][j];
    }
  }

  return matrix;
}

std::vector<std::string> DataLoader::listFiles(const std::string &path) {
  std::vector<std::string> local_files;
  int64_t file_count = 0;
  std::vector<int64_t> file_ranks;
  DIR *dir;
  struct dirent *entry;

  MPIController &global_controller = globalController();
  int mpi_rank = global_controller.mpiRank();
  int mpi_size = global_controller.mpiSize();

  if (mpi_rank == 0) {
    dir = opendir(path.c_str());
    if (dir == nullptr) {
      std::cerr << "Could not open path: " << path << std::endl;
      return local_files;
    }

    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_type == DT_REG && entry->d_name[0] != '.') {
        file_ranks.emplace_back(file_count);
        file_count++;
      }
    }
    closedir(dir);
  }

  if (globalParallelismMode() == DATA_PARALLELISM) {
    int counts[mpi_size];
    if (mpi_rank == 0) {
      for (int i = 0; i < mpi_size; i++) {
        counts[i] = file_count / mpi_size;
      }

      int remainder = file_count % mpi_size;

      for (int i = 0; i < remainder; i++) {
        counts[i]++;
      }
    }

    int local_count;

    global_controller.mpiScatter(counts, 1, local_count, 1, 0);

    std::vector<int64_t> local_file_ranks(local_count);
    int ret = global_controller.mpiScatterv(file_ranks, counts, local_file_ranks,
                                            local_count, 0);

    for (auto &rank : local_file_ranks) {
      local_files.emplace_back(path + "part-" + formatString(rank));
    }
  } else {
    global_controller.mpiBcast(file_count, 0);

    if (mpi_rank != 0) {
        file_ranks.resize(file_count);
    }
    global_controller.mpiBcast(file_ranks, file_count, 0);
    std::vector<int64_t>& local_file_ranks = file_ranks;

    for (auto &rank : local_file_ranks) {
      local_files.emplace_back(path + "part-" + formatString(rank));
    }
  }


  return local_files;
}
