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

  void initGlobalGrads() {
    const int layers_num = _layers_size.size();
    _global_grads.reserve(layers_num);
  }

  std::vector<int> getDoneRanks() { return _done_ranks; }

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

  void setDoneStatus(const bool &done) { _done = done; }

  bool getDoneStatus() { return _done; }

  Eigen::MatrixXf &getWeights(const int &layer_rank) {
    return _global_weigths[layer_rank];
  }

  Eigen::MatrixXf &getBias(const int &layer_rank) {
    return _global_bias[layer_rank];
  }

  Eigen::MatrixXf &getGrads(const int &layer_rank) {
    return _global_grads[layer_rank];
  }

private:
  std::vector<Eigen::MatrixXf> _global_weigths;
  std::vector<Eigen::MatrixXf> _global_bias;
  std::vector<Eigen::MatrixXf> _global_grads;
  std::vector<int>
      _done_ranks; // master永远最后完成（每个part样本量一样的情况下）
  bool _done = false;
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
