/**
 * Sequential model class implementation
 */

#include "Sequential.hpp"

#include <iostream>

using namespace DeepLearningFramework;

Sequential::Sequential(std::vector<Module *> &model, Losses::MSE loss) {
  _model = model;
  _loss = loss;
}

void Sequential::forward(Eigen::MatrixXf &x) {
  std::vector<Module *>::iterator it;

  for (it = _model.begin(); it != _model.end(); it++)
    (*it)->forward(x, x);
}

void Sequential::backward(float &loss, const Eigen::MatrixXf &y,
                          Eigen::MatrixXf &y_pred) {
  // calculate loss
  _loss.forward(loss, y, y_pred);

  // back propagation
  Eigen::MatrixXf lossDerivative;
  _loss.backward(lossDerivative, y, y_pred);
  for (auto it = _model.rbegin(); it != _model.rend(); it++) {
    (*it)->backward(lossDerivative, lossDerivative);
  }
}

void Sequential::setLR(float lr) {
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    (*it)->setLR(lr);
}

uint32_t Sequential::getParametersCount() {
  uint32_t parametersCount = 0;
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    parametersCount += (*it)->getParametersCount();
  return parametersCount;
}

void Sequential::printDescription() {
  // layer description
  std::cout << "Model:" << std::endl;
  std::vector<Module *>::iterator it;
  for (it = _model.begin(); it != _model.end(); it++)
    (*it)->printDescription();

  // loss
  std::cout << "\nWith loss:" << std::endl;
  _loss.printDescription();

  // parameters count
  std::cout << "\nNumber of parameters:" << this->getParametersCount()
            << std::endl;
}
