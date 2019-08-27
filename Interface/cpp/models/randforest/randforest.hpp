#pragma once
// #include "base.hpp"
#include <vector>
#include "base/base_template.hpp"

namespace models {
namespace rsds {
namespace carpoint {

class CarPoint : public base_template::RandForest<std::vector<std::vector<double>>, std::vector<double>> {
public:
    CarPoint();
    ~CarPoint();

private:
    void collect_pred(const std::vector<double> &input, std::vector<double> &classes) override;
};

}  // namespace carpoint
}  // namespace rsds
}  // namepsace models
