#include "DataLoader.hpp"
#include "Linear.hpp"
#include "MSE.hpp"
#include "Metrics.hpp"
#include "ReLU.hpp"
#include "Sequential.hpp"
#include "Softmax.hpp"
#include "Trainer.hpp"
#include "GlobalState.hpp"
#include "Common.hpp"

using namespace DeepLearningFramework;

int main() {
  std::vector<int> layers_size = {2, 10, 10, 2};
  initialize(layers_size);
  /* Model creation */
  std::vector<Module *> layers;
  const int layers_num = layers_size.size();
  for (int i = 1; i < layers_num; ++i) {
    layers.emplace_back(new Layers::Linear(layers_size[i - 1], layers_size[i]));
    if (i == layers_num - 1)
      layers.emplace_back(new Activations::Softmax());
    else {
      layers.emplace_back(new Activations::ReLU());
    }
  }

  /* Generate train and test sets */
  Eigen::MatrixXf y_train, X_train, y_test, X_test;
  // std::string data_path = "../data/iris/";
  std::string data_path = "../data/uniform_sample_size_per_part/";
  DataLoader::load(data_path, X_train, y_train,
                   X_test, y_test);

  Losses::MSE mseLoss;

  Sequential model(layers, mseLoss);
  // model.printDescription();

  /* Train params */
  float lr = 0.002f;
  model.setLR(lr);
  uint32_t epochs = 200, step = 1;
  constexpr auto batch_size = 64;
  constexpr auto feature_dim = 2;

  // number of train and test samples
  std::vector<float> train_acc, test_acc;

  // Train model
  Trainer::trainModel<batch_size, feature_dim>(train_acc, test_acc, model, epochs, y_train,
                                  X_train, y_test, X_test, step);

  finalize();
}
