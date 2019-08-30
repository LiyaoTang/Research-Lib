#pragma once
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../utils/anchor.hpp"
#include "unit_base/unit_test_base.hpp"

namespace test {
class Anchor_Test : public UnitTestBase {
public:
    void test() {
        std::cout << " === Anchor Test === \n";
        // std::cout << std::fixed << std::setprecision(6);

        int stride = 8;
        std::vector<double> ratios{0.5, 1, 1.5};
        std::vector<double> scales{2};

        std::cout << "stride= " << stride << "; scales= ";
        for (auto& it : scales) {
            std::cout << it << " ";
        }
        std::cout << "; ratios= ";
        for (auto& it : ratios) {
            std::cout << it << " ";
        }
        std::cout << std::endl;

        utils::Anchor anchor(stride, ratios, scales);
        cv::Mat anchor_volume = anchor.generate_anchors_volume(2, std::pair<int, int>(2, 2));
        std::cout << "anchor_volume: \n"
                  << anchor_volume << "\n";
    }
};
}