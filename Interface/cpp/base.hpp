#ifndef BASE_H
#define BASE_H

#include <vector>

namespace models{
namespace base{

/** 
 * base class for all models 
 * **/
class Model{
public:
    const int class_num = 0;
    const int feature_len = 0;

    Model(int class_num, int feature_len);
    ~Model();

    virtual std::vector<double> pred_prob(std::vector<double> &input); // virtual => dynamic
    virtual std::vector<std::vector<double>> *pred_prob(std::vector<std::vector<double>> &input);
};


/** 
 * base class for random forest model 
 * **/
class RandForest : public Model{
public:
    const int tree_num = 0;

    RandForest(int class_num, int feature_len, int tree_num=0);
    ~RandForest();

    std::vector<double> pred_prob(std::vector<double> &input) override;
    std::vector<std::vector<double>> *pred_prob(std::vector<std::vector<double>> &input) override;

private:
    virtual std::vector<double> collect_pred(const std::vector<double> &input) = 0;
};

} // namespace base
} // namepsace models
#endif