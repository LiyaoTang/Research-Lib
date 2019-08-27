#include "base_template.hpp"
#include <glog/logging.h>
#include <iostream>

namespace models {
namespace base_template {

/** 
 * definition of Model
 * **/
template <typename Batch_T, typename Input_T>
Model<Batch_T, Input_T>::Model(int class_num, int feature_len) : class_num(class_num), feature_len(feature_len) {}
template <typename Batch_T, typename Input_T>
Model<Batch_T, Input_T>::~Model() {}

/**
 * definition of RandForest
 * **/
template <typename Batch_T, typename Input_T>
RandForest<Batch_T, Input_T>::RandForest(int class_num, int feature_len, int tree_num = 0) : model(class_num, feature_len), tree_num(tree_num) {}
template <typename Batch_T, typename Input_T>
RandForest<Batch_T, Input_T>::~RandForest() {}

template <typename Batch_T, typename Input_T>  // predict single input
Input_T RandForest<Batch_T, Input_T>::pred_prob(Input_T &input) {
    // chk feature len
    if (input.size() != this->feature_len) {
        std::ostringstream err_string;
        err_string << "expected input example with length %d but get %d", input.size(), this->feature_len;
        LOG(WARNING) << err_string.str();
        throw std::runtime_error(err_string.str());
    }

    // get class prediction into a Input_T
    Input_T class_pred(this->collect_pred(input));

    // normalize to prob
    double sum_pred = 0;
    for (auto &pred : class_pred) {
        sum_pred += pred;
    }
    for (auto &pred : class_pred) {
        pred /= sum_pred;
    }
    return class_pred;
}

template <typename Batch_T, typename Input_T>  // predict batch input
std::shared_ptr<Batch_T> RandForest<Batch_T, Input_T>::pred_prob(Batch_T &input, int batch_size) {
    Batch_T *prediction = new Batch_T(input);  // copy constructor
    this->pred_prob(input, batch_size, prediction);
    return prediction;
}

}  // namespace base_template
}  // namespace models