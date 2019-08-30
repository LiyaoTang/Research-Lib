#pragma once
// #include "base.hpp"
#include <vector>
#include "../base/base_template.hpp"

namespace models {
namespace rsds {
namespace carpoint {

using Batch_T = std::vector<std::vector<double>>;
using Input_T = std::vector<double>;
class CarPoint : public base_template::RandForest<Batch_T, Input_T> {
public:
    CarPoint();
    ~CarPoint();

private:
    void collect_pred(const Input_T &input, Input_T &classes) override;
};

}  // namespace carpoint
}  // namespace rsds
}  // namepsace models
