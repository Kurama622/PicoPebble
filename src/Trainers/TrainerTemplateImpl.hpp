/**
 * Template function from Trainer class implementation
 */

#include "Common.hpp"
#include <chrono>
#include <cmath>
#include <thread>

using namespace DeepLearningFramework;

inline Eigen::MatrixXf oneHotEncoding(const Eigen::MatrixXf& input) {
    int n = input.rows(); // 输入矩阵的行数
    int N_classes = input.maxCoeff() + 1;
    Eigen::MatrixXf one_hot = Eigen::MatrixXf::Zero(n, N_classes);
    for (int i = 0; i < n; ++i) {
        one_hot(i, static_cast<int>(input(i))) = 1.0;
    }
    return one_hot;
}

template <uint32_t batch_size, uint32_t feature_dim>
void Trainer::trainModel(std::vector<float> train_acc,
                         std::vector<float> test_acc, Sequential &model,
                         uint32_t epochs, const Eigen::MatrixXf &y_train,
                         const Eigen::MatrixXf &X_train,
                         const Eigen::MatrixXf &y_test,
                         const Eigen::MatrixXf &X_test, uint32_t step) {

  uint32_t batch_num = X_train.rows() / batch_size;

  // set the flag for stopping synchronization parameters.
  trainFinishFlag().setStatus(epochs - 1, batch_num - 1);

  for (uint32_t i = 0; i < epochs; i++) {
    float loss = 0.f;
    for (uint32_t batch_idx = 0; batch_idx < batch_num; batch_idx++) {
      float batch_loss = 0.f;
      globalTrainStatus().setStatus(i, batch_idx);
      Eigen::MatrixXf X_batch =
          X_train.block<batch_size, feature_dim>(batch_idx * batch_size, 0);
      Eigen::MatrixXf y_batch =
          y_train.block<batch_size, 1>(batch_idx * batch_size, 0);
      y_batch = oneHotEncoding(y_batch);

      model.forward(X_batch);
      model.backward(batch_loss, y_batch, X_batch);
      loss += batch_loss;
    }

    addAccuracy(train_acc, model, y_train, X_train);
    addAccuracy(test_acc, model, y_test, X_test);

    loss /= batch_num;

    if (i % step == 0)
      std::cout << "Rank: " << globalController().mpiRank() << ", Epoch: " << i
                << ", train accuracy: " << train_acc.at(i) << ", loss: " << loss
                << ", test accuracy: " << test_acc.at(i) << std::endl;
  }
}
