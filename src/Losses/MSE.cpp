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
