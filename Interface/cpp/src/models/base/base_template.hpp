#pragma once
// #include <glog/logging.h>
#include <iostream>
#include <memory>
#include <sstream>
#include "base_template.hpp"

namespace models {
namespace base_template {

/** 
 * base class template for all models 
 * **/
template <typename Batch_T, typename Input_T>
class Model {
public:
    const int class_num   = 0;
    const int feature_len = 0;

    Model(int class_num, int feature_len) : class_num(class_num),
                                            feature_len(feature_len) {}

    virtual Input_T pred_prob(Input_T &input) = 0;  // pure virtual => abstract class
    virtual std::shared_ptr<Batch_T> pred_prob(Batch_T &input, int batch_size) = 0;
};

/** 
 * base class template for random forest model
 * **/
template <typename Batch_T, typename Input_T>
class RandForest : private Model<Batch_T, Input_T> {
public:
    const int tree_num    = 0;
    const int class_num   = 0;
    const int feature_len = 0;

    RandForest(int class_num, int feature_len, int tree_num = 0) : Model<Batch_T, Input_T>(class_num, feature_len),
                                                                   tree_num(tree_num) {}

    Input_T pred_prob(Input_T &input) override;
    std::shared_ptr<Batch_T> pred_prob(Batch_T &input, int batch_size) override;

private:
    virtual void collect_pred(const Input_T &input, Input_T &classes) = 0;
};

/**
 * definition of RandForest
 * **/
template <typename Batch_T, typename Input_T>  // predict single input
Input_T RandForest<Batch_T, Input_T>::pred_prob(Input_T &input) {
    // chk feature len
    if (input.size() != this->feature_len) {
        std::ostringstream err_string;
        err_string << "expected input example with length %d but get %d", input.size(), this->feature_len;
        // LOG(WARNING) << err_string.str();
        throw std::runtime_error(err_string.str());
    }

    // get class prediction into a Input_T
    Input_T class_pred;
    this->collect_pred(input, class_pred);

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
    std::shared_ptr<Batch_T> pred = std::make_shared<Batch_T>(input);  // copy constructor
    for (int i = 0; i < batch_size; i++) {
        (*pred)[i] = pred_prob(input[i]);
    }
    return pred;
}

}  // namespace base_template
}  // namespace models