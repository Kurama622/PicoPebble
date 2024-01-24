/**
 * Softmax activation class implementation
 */

#include "Softmax.hpp"

#include <iostream>

using namespace DeepLearningFramework::Activations;

Softmax::Softmax() {}

void Softmax::forward(Eigen::MatrixXf &out, const Eigen::MatrixXf &x) {
  Softmax::equation(out, x);
  _forward_input_with_softmax_applied = out;
}

void Softmax::backward(Eigen::MatrixXf &din, const Eigen::MatrixXf &dout) {

  const Eigen::MatrixXf grad = dout;

  for (int i = 0; i < dout.rows(); ++i) {
    for (int j = 0; j < dout.cols(); ++j) {
      for (int k = 0; k < dout.cols(); ++k) {
        if (j == k) {
          din(i, j) += grad(i, k) *
                         _forward_input_with_softmax_applied(i, k) *
                         (1.f - _forward_input_with_softmax_applied(i, j));
        } else {
          din(i, j) += grad(i, k) *
                         _forward_input_with_softmax_applied(i, k) *
                         (-_forward_input_with_softmax_applied(i, j));
        }
      }
    }
  }

}

void Softmax::printDescription() {
  std::cout << "Softmax activation" << std::endl;
}

void Softmax::equation(Eigen::MatrixXf &y, const Eigen::MatrixXf &x) {
  Eigen::MatrixXf exp_x = x.array().exp();
  y = x;
  for (int row = 0; row < x.rows(); ++row) {
    y.row(row) = exp_x.row(row) / exp_x.row(row).sum();
  }
}
