#pragma once
#include <memory>
#include <vector>

namespace models {
namespace base {

/** 
 * base class for all models 
 * **/
class Model {
public:
    const int pred_len    = 0;
    const int feature_len = 0;

    Model(int pred_len, int feature_len);
    ~Model();

    virtual void pred_prob(std::vector<double> &input, std::vector<double> &output);  // virtual => dynamic
    virtual std::vector<double> pred_prob(std::vector<double> &input);
    virtual void pred_prob(std::vector<std::vector<double>> &input, std::vector<std::vector<double>> &output);
};

/** 
 * base class for random forest model 
 * **/
class RandForest : public Model {
public:
    const int tree_num  = 0;
    const int class_num = 0;

    RandForest(int pred_len, int feature_len, int tree_num = 0);
    ~RandForest();

    void pred_prob(std::vector<double> &input, std::vector<double> &output) override;

private:
    virtual void collect_pred(const std::vector<double> &input, std::vector<double> &classes) = 0;
};

// /** 
//  * base class for tracking model 
//  * **/
// class Tracker {
// public:
//     const int tree_num  = 0;
//     const int class_num = 0;

//     Tracker(int pred_len, int feature_len, int tree_num = 0);
//     ~Tracker();

//     virtual std::shared_ptr<std::vector<double>> pred_prob(std::vector<double> &input);
//     virtual void pred_prob(std::vector<std::vector<double>> &input, std::vector<std::vector<double>> &output);

// private:
//     virtual void collect_pred(const std::vector<double> &input, std::vector<double> &classes) = 0;
// };

}  // namespace base
}  // namepsace models