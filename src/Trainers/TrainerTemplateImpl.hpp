/**
 * Template function from Trainer class implementation
 */

#include "Common.hpp"
#include <chrono>
#include <cmath>
#include <thread>

using namespace DeepLearningFramework;

template <uint32_t batch_size>
void Trainer::trainModel(std::vector<float> train_acc,
                         std::vector<float> test_acc, Sequential &model,
                         uint32_t epochs, const Eigen::MatrixXf &y_train,
                         const Eigen::MatrixXf &X_train,
                         const Eigen::MatrixXf &y_test,
                         const Eigen::MatrixXf &X_test, uint32_t step) {

  uint32_t batch_num = X_train.rows() / batch_size;
  std::cout << "batch_num: " << batch_num << std::endl;

  // set the flag for stopping synchronization parameters.
  if (globalSyncStep()) {
    uint32_t finish_epoch_idx = 0;
    finish_epoch_idx =
        (epochs * batch_num) - (epochs * batch_num) % globalSyncStep();
    if (finish_epoch_idx % batch_num) {
      finish_epoch_idx = finish_epoch_idx / batch_num;
    } else {
      finish_epoch_idx = finish_epoch_idx / batch_num - 1;
    }

    uint32_t finish_batch_idx =
        batch_num - 1 - (epochs * batch_num) % globalSyncStep() % batch_num;
    trainFinishFlag().setStatus(finish_epoch_idx, finish_batch_idx);
  } else {
    trainFinishFlag().setStatus(epochs - 1, batch_num - 1);
  }

  for (uint32_t i = 0; i < epochs; i++) {
    float loss = 0.f;
    for (uint32_t batch_idx = 0; batch_idx < batch_num; batch_idx++) {
      float batch_loss = 0.f;
      globalTrainStatus().setStatus(i, batch_idx);
      Eigen::MatrixXf X_batch =
          X_train.block<batch_size, 2>(batch_idx * batch_size, 0);
      Eigen::MatrixXf y_batch =
          y_train.block<batch_size, 2>(batch_idx * batch_size, 0);

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
