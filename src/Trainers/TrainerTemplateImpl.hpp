/**
 * Template function from Trainer class implementation
 */

using namespace DeepLearningFramework;

template <uint32_t batch_size>
void Trainer::trainModel(std::vector<float> train_acc,
                           std::vector<float> test_acc, Sequential &model,
                           uint32_t epochs, const Eigen::MatrixXf &y_train,
                           const Eigen::MatrixXf &X_train,
                           const Eigen::MatrixXf &y_test,
                           const Eigen::MatrixXf &X_test, uint32_t step) {

  uint32_t batch_num = X_train.rows() / batch_size;

  for (uint32_t i = 0; i < epochs; i++) {
    float loss = 0.f;
    for (uint32_t batch_idx = 0; batch_idx < batch_num; batch_idx++) {
      float batch_loss = 0.f;
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
      std::cout << "Epoch: " << i << ", train accuracy: " << train_acc.at(i)
                << ", loss: " << loss
                << ", test accuracy: " << test_acc.at(i) << std::endl;
  }
}
