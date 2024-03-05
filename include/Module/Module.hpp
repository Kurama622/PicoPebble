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
