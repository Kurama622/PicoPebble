#include "DataLoader.hpp"
#include <dirent.h>
#include <fstream>
#include <iostream>

using namespace DeepLearningFramework;

void DataLoader::load(const std::string &path, Eigen::MatrixXf &X_train,
                      Eigen::MatrixXf &y_train, Eigen::MatrixXf &X_test,
                      Eigen::MatrixXf &y_test) {
  std::string X_train_path = path + "train_features/";
  std::string y_train_path = path + "train_labels/";
  std::string X_test_path = path + "test_features/";
  std::string y_test_path = path + "test_labels/";
  load_matrix(X_train_path, X_train);
  load_matrix(y_train_path, y_train);
  load_matrix(X_test_path, X_test);
  load_matrix(y_test_path, y_test);
}

void DataLoader::load_matrix(const std::string &path,
                             Eigen::MatrixXf &concat_matrix) {
  std::vector<std::string> part_files = listFiles(path);

  for (int i = 0; i < part_files.size(); i++) {
    std::cout << "Reading file: " << part_files[i] << std::endl;
    Eigen::MatrixXf matrix = readMatrixFromFile(part_files[i]);
    if (concat_matrix.rows() == 0) {
      concat_matrix.resize(matrix.rows() * part_files.size(), matrix.cols());
    }
    concat_matrix.block(matrix.rows() * i, 0, matrix.rows(), matrix.cols()) =
        matrix;
  }

  std::cout << "concat Matrix:" << std::endl << concat_matrix << std::endl;
}

Eigen::MatrixXf DataLoader::readMatrixFromFile(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Could not open the file: " << filename << std::endl;
    return Eigen::MatrixXf();
  }

  std::string line;
  std::vector<std::vector<float>> data;
  while (std::getline(file, line)) {
    std::vector<float> row;
    std::stringstream line_stream(line);
    float value;
    while (line_stream >> value) {
      row.push_back(value);
      if (line_stream.peek() == ',') {
        line_stream.ignore();
      }
    }
    data.push_back(row);
  }

  Eigen::MatrixXf matrix(data.size(), data[0].size());
  for (size_t i = 0; i < data.size(); ++i) {
    for (size_t j = 0; j < data[i].size(); ++j) {
      matrix(i, j) = data[i][j];
    }
  }

  return matrix;
}

std::vector<std::string> DataLoader::listFiles(const std::string &path) {
  std::vector<std::string> files;
  DIR *dir;
  struct dirent *entry;

  dir = opendir(path.c_str());
  if (dir == nullptr) {
    std::cerr << "Could not open path: " << path << std::endl;
    return files;
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG && entry->d_name[0] != '.') {
      std::string filename = entry->d_name;
      if (filename.rfind("part", 0) == 0) {
        files.push_back(path + filename);
      }
    }
  }

  closedir(dir);
  return files;
}
