#pragma once

namespace models {
namespace base_template {

/** 
 * base class template for all models 
 * **/
template <typename Batch_T, typename Input_T>
class Model {
private:
public:
    const int class_num   = 0;
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
    const int tree_num    = 0;
    const int class_num   = 0;
    const int feature_len = 0;

    RandForest(int class_num, int feature_len, int tree_num = 0);
    ~RandForest();

    Input_T pred_prob(Input_T &input) override;
    Batch_T *pred_prob(Batch_T &input, int batch_size) override;
    void pred_prob(Batch_T &input, int batch_size, Batch_T *pred) override;
};

}  // namespace base_template
}  // namespace models