#include <iostream>
#include <sstream>
#include <vector>
#include <glog/logging.h>
#include "base.hpp"

namespace models{
namespace base{

/**
 * definition of the base class for all models
 * **/
Model::Model(int class_num, int feature_len): class_num(class_num), feature_len(feature_len) {}
Model::~Model() {}

std::vector<double> Model::pred_prob(std::vector<double> &input){
    throw std::runtime_error("function not implemented");
}
std::vector<std::vector<double>> *Model::pred_prob(std::vector<std::vector<double>> &input){
    throw std::runtime_error("function not implemented");
}


/** 
 * definition of the base class for RandForest 
 * **/
RandForest::RandForest(int class_num, int feature_len, int tree_num=0): Model(class_num, feature_len), tree_num(tree_num) {}
RandForest::~RandForest() {}

// predict single input example
std::vector<double> RandForest::pred_prob (std::vector<double> &features){
    // chk feature len
    if (features.size() != this->feature_len){
        std::stringstream err_string;
        err_string << "expected features with length %d but get %d", features.size(), this->feature_len;
        LOG(WARNING) << err_string.str();
        throw std::runtime_error(err_string.str());
    }

    // get class prediction into vector
    std::vector<double> class_pred = this->collect_pred(features);
    
    // normalize to prob
    double cnt_sum = 0;
    for (auto &cnt : class_pred){
        cnt_sum += cnt;
    }
    for (auto &cnt : class_pred){
        cnt /= cnt_sum;
    }
    return class_pred;
}

// predict batch input
std::vector<std::vector<double>> *RandForest::pred_prob(std::vector<std::vector<double>> &input){
    // initialize with 0
    using batch_T = std::vector<std::vector<double>>;
    batch_T *prediction = new batch_T(input.size(), std::vector<double>(this->feature_len));

    for (int i = 0; i < input.size(); i++){
        (*prediction)[i] = this->pred_prob(input[i]);
    }
    return prediction;
}


} // namespace base
} // namepsace models