#pragma once

#include "GlobalState.hpp"
#include "mpi/MpiController.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

namespace DeepLearningFramework {

class GlobalState {
public:
  void setLayersSize(const std::vector<int> &layers_size) {
    _layers_size = layers_size;
  }

  std::vector<int> getLayersSize() { return _layers_size; }

  int getLayersNum() { return _layers_size.size(); }

  void initGlobalWeights() {
    const int layers_num = _layers_size.size();
    for (int i = 1; i < layers_num; ++i) {
      _global_weigths.emplace_back(
          Eigen::MatrixXf::Random(_layers_size[i - 1], _layers_size[i]));
    }
  }

  void initGlobalBias() {
    const int layers_num = _layers_size.size();
    for (int i = 1; i < layers_num; ++i) {
      _global_bias.emplace_back(Eigen::MatrixXf::Random(1, _layers_size[i]));
    }
  }

  void setWeights(Eigen::MatrixXf &weights, const int &layer_rank) {
    _global_weigths[layer_rank] = weights;
  }

  void setBias(Eigen::MatrixXf &bias, const int &layer_rank) {
    _global_bias[layer_rank] = bias;
  }

  void setGlobalBias() {
    const int layers_num = _layers_size.size();
    for (int i = 1; i < layers_num; ++i) {
      _global_bias.emplace_back(Eigen::MatrixXf::Random(1, _layers_size[i]));
    }
  }

  Eigen::MatrixXf &getWeights(const int &layer_rank) {
    return _global_weigths[layer_rank];
  }

  Eigen::MatrixXf &getBias(const int &layer_rank) {
    return _global_bias[layer_rank];
  }

private:
  std::vector<Eigen::MatrixXf> _global_weigths;
  std::vector<Eigen::MatrixXf> _global_bias;
  std::vector<int> _layers_size;
};

inline MPIController &globalController() {
  static MPIController mpi_controller;
  return mpi_controller;
}

inline GlobalState &globalState() {
  static GlobalState global_state;
  return global_state;
}

} // namespace DeepLearningFramework
