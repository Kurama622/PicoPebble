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
