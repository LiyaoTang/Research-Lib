#include "base.hpp"
// #include <glog/logging.h>
#include <iostream>
#include <sstream>
#include <vector>

namespace models {
namespace base {

/**
 * definition of the base class for all models
 * **/
Model::Model(int pred_len, int feature_len) : pred_len(pred_len),
                                              feature_len(feature_len) {}
Model::~Model() {}

void Model::pred_prob(std::vector<double> &input, std::vector<double> &output) {
    output.resize(this->pred_len);
    throw std::runtime_error("function not implemented");
}
std::vector<double> Model::pred_prob(std::vector<double> &input) {
    std::vector<double> output(this->pred_len, 0);
    this->pred_prob(input, output);
    return output;
}
void Model::pred_prob(std::vector<std::vector<double>> &input, std::vector<std::vector<double>> &output) {
    output.resize(input.size());
    for (int i = 0; i < input.size(); i++) {
        this->pred_prob(input[i], output[i]);
    }
}

/** 
 * definition of the base class for RandForest 
 * **/
RandForest::RandForest(int class_num, int feature_len, int tree_num) : Model(class_num, feature_len),
                                                                       tree_num(tree_num) {}
RandForest::~RandForest() {}

// predict single input example
void RandForest::pred_prob(std::vector<double> &input, std::vector<double> &output) {
    output.resize(this->pred_len);
    std::fill(output.begin(), output.end(), 0);

    // chk feature len
    if (input.size() != this->feature_len) {
        std::stringstream err_string;
        err_string << "expected features with length %d but get %d", input.size(), this->feature_len;
        // LOG(ERROR) << err_string.str();
        throw std::runtime_error(err_string.str());
    }

    // get class prediction into vector
    this->collect_pred(input, output);

    // normalize to prob
    double cnt_sum = 0;
    for (auto &cnt : output) {
        cnt_sum += cnt;
    }
    for (auto &cnt : output) {
        cnt /= cnt_sum;
    }
    return;
}

// predict batch input

}  // namespace base
}  // namepsace models