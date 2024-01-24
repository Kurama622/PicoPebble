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
