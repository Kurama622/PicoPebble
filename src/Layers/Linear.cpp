/**
 * Linear layer class implementation
 */

#include "Linear.hpp"
#include "GlobalState.hpp"

#include <iostream>

using namespace DeepLearningFramework::Layers;

int Linear::_layer_count = 0;
Linear::Linear(int input_size, int output_size) {
  _input_size = input_size;
  _output_size = output_size;
  _layer_rank = _layer_count++;
  _weights = &globalState().getWeights(_layer_rank);
  _bias = &globalState().getBias(_layer_rank);
}

void Linear::forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) {
  _forward_input = x;
  out = x.matrix() * _weights->matrix();
  for (int row = 0; row < out.rows(); ++row) {
    out.row(row).array() -= _bias->array();
  }
}

void Linear::backward(Eigen::MatrixXf &din, const Eigen::MatrixXf &dout) {
  // update weights and bias
  (*_weights) -= _lr * (_forward_input.transpose() * dout);
  (*_bias) -= _lr * dout.colwise().mean();

  // Calculate the gradient component for each input, which corresponds to the
  // output of the previous layer.
  din = dout * _weights->transpose();
}

void Linear::printDescription() {
  std::cout << "Linear Layer [" << _input_size << ", " << _output_size << "], "
            << "parameters: " << this->getParametersCount()
            << ", learning rate: " << _lr << std::endl;
}

void Linear::setLR(float lr) { _lr = lr; }

uint32_t Linear::getParametersCount() {
  return _input_size * _output_size + _output_size;
}

Eigen::MatrixXf Linear::getWeights() { return *_weights; }

Eigen::MatrixXf Linear::getBias() { return *_bias; }

void Linear::setWeightsAndBias(const Eigen::MatrixXf &weights,
                               const Eigen::MatrixXf &bias) {
  *_weights = weights;
  *_bias = bias;
}
