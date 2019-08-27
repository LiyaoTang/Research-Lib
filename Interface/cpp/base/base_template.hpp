#pragma once
#include <memory>

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

    Model(int class_num, int feature_len);
    ~Model();

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

    RandForest(int class_num, int feature_len, int tree_num = 0);
    ~RandForest();

    Input_T pred_prob(Input_T &input) override;
    std::shared_ptr<Batch_T> pred_prob(Batch_T &input, int batch_size) override;

private:
    virtual Input_T &collect_pred(const Input_T &input) = 0;
};

}  // namespace base_template
}  // namespace models