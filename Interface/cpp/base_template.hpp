#ifndef BASE_TEMPLATE_H
#define BASE_TEMPLATE_H

#include <glog/logging.h>
#include <iostream>

namespace models {
namespace base_template {

/** 
 * base class template for all models 
 * **/
template <typename Batch_T, typename Input_T>
class Model {
private:
public:
    const int class_num = 0;
    const int feature_len = 0;

    Model(int class_num, int feature_len);
    ~Model();

    virtual Input_T pred_prob(Input_T &input) = 0;  // pure virtual => abstract class
    virtual Batch_T *pred_prob(Batch_T &input, int batch_size) = 0;
    virtual void pred_prob(Batch_T &input, int batch_size, Batch_T *pred) = 0;
};

template <typename Batch_T, typename Input_T>
Model<Batch_T, Input_T>::Model(int class_num, int feature_len) : class_num(class_num), feature_len(feature_len) {}
template <typename Batch_T, typename Input_T>
Model<Batch_T, Input_T>::~Model() {}

/** 
 * base class template for random forest model
 * **/
template <typename Batch_T, typename Input_T>
class RandForest : private Model<Batch_T, Input_T> {
private:
    virtual Input_T &collect_pred(const Input_T &input) = 0;

public:
    const int tree_num = 0;
    const int class_num = 0;
    const int feature_len = 0;

    RandForest(int class_num, int feature_len, int tree_num = 0);
    ~RandForest();

    Input_T pred_prob(Input_T &input) override;
    Batch_T *pred_prob(Batch_T &input, int batch_size) override;
    void pred_prob(Batch_T &input, int batch_size, Batch_T *pred) override;
};

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
Batch_T *RandForest<Batch_T, Input_T>::pred_prob(Batch_T &input, int batch_size) {
    Batch_T *prediction = new Batch_T(input);  // copy constructor
    this->pred_prob(input, batch_size, prediction);
    return prediction;
}

template <typename Batch_T, typename Input_T>  // predict batch input with pre-allocated pred
void RandForest<Batch_T, Input_T>::pred_prob(Batch_T &input, int batch_size, Batch_T *pred) {
    for (int idx = 0, idx < batch_size, idx++) {
        (*pred)[idx] = this->pred_prob(input[i]);
    }
}

}  // namespace base_template
}  // namespace models
#endif