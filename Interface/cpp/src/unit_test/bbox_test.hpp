#pragma once
#include <iostream>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <vector>
#include "../utils/bbox.hpp"
#include "unit_base/unit_test_base.hpp"

namespace test {

class Bbox_Test : public UnitTestBase {
public:
    void test() {
        std::cout << " === Bbox Test === \n";
        std::vector<int> box_vec = {1, 2, 3, 4};
        box_vec.shrink_to_fit();
        cv::Mat box_cv(1, 4, CV_32S);
        for (int i = 0; i < box_vec.size(); i++) {
            box_cv.at<int>((0, i)) = box_vec[i];
        }
        // cv::Mat box_cv(box_vec, false);

        box_cv = cv::repeat(box_cv, 3, 1);
        std::ostringstream oss;

        std::copy(box_vec.begin(), box_vec.end() - 1, std::ostream_iterator<int>(oss, ","));
        oss << box_vec.back();
        std::cout << "input xywh box_vec = " << oss.str() << std::endl;
        std::cout << "input xywh box_cv = \n"
                  << box_cv << std::endl;

        std::cout << " --- convert xywh -> xyxy --- \n";
        oss.str("");
        auto rst_vec = utils::bbox::xywh_to_xyxy(box_vec);  // test vec
        std::copy(rst_vec.begin(), rst_vec.end() - 1, std::ostream_iterator<int>(oss, ","));
        oss << rst_vec.back();
        std::cout << "rst_vec = " << oss.str() << std::endl;
        auto rst_cv = utils::bbox::xywh_to_xyxy(box_cv);  // test cv mat
        std::cout << "rst_cv = \n"
                  << rst_cv << std::endl;

        std::cout << " --- convert xyxy -> xywh --- \n";
        oss.str("");
        auto revert_vec = utils::bbox::xyxy_to_xywh(rst_vec);  // test vec
        std::copy(revert_vec.begin(), revert_vec.end() - 1, std::ostream_iterator<int>(oss, ","));
        oss << revert_vec.back();
        std::cout << "revert_vec = " << oss.str() << std::endl;
        auto revert_cv = utils::bbox::xyxy_to_xywh(rst_cv);  // test cv mat
        std::cout << "revert_cv = \n"
                  << revert_cv << std::endl;
    }
};
}