#ifndef RANDFOREST_H
#define RANDFOREST_H

#include "base.hpp"
// #include "base_template.hpp"
#include <vector>

namespace models{
namespace rsds{
namespace carpoint{

class CarPoint : public base::RandForest {
public:
    CarPoint();
    ~CarPoint();

private:
    std::vector<double> collect_pred(const std::vector<double> &input) override;
};

} // namespace carpoint
} // namespace rsds
} // namepsace models
#endif