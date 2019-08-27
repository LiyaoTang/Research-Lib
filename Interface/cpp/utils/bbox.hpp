#pragma once
#include <cmath>
#include <vector>
#include <opencv2/core.hpp>

namespace utils {
class bbox {
public:
    static std::vector<int> xyxy_to_xywh(int x1, int y1, int x2, int y2) {
        int cx = (x1 + x2) / 2;
        int cy = (y1 + y2) / 2;
        int w  = x2 - x1;
        int h  = y2 - y1;
        return std::vector{cx, cy, w, h};
    }
    static std::vector<int> xyxy_to_xywh(std::vector<int> xyxy) {
        int cx = (int)std::floor((xyxy[0] + xyxy[2]) / 2.0);
        int cy = (int)std::floor((xyxy[1] + xyxy[3]) / 2.0);
        int w  = xyxy[2] - xyxy[0];
        int h  = xyxy[3] - xyxy[1];
        return std::vector{cx, cy, w, h};
    }
    static cv::Mat xyxy_to_xywh(cv::Mat xyxy){
        cv::Mat xywh(xyxy.size(), CV_64F);
        cv::Mat half(xyxy.rows, 1, CV_64F, cv::Scalar(0.5));
        cv::Mat mat_arr[] = {(xyxy.col(0) + xyxy.col(2)).mul(half),
                             (xyxy.col(1) + xyxy.col(3)).mul(half),
                             xyxy.col(2) - xyxy.col(0),
                             xyxy.col(3) - xyxy.col(1)};
        cv::hconcat(mat_arr, 4, xyxy);

        cv::Mat xyxy_int(xywh.size(), CV_32S); // convert to int
        auto src_it = xyxy.begin<double>();
        auto dst_it = xyxy_int.begin<int>();
        for (; src_it != xyxy.end<double>; src_it ++, dst_it ++){
            (*dst_it) = (int)std::ceil(*src_it);
        }
        return xyxy_int;
    }

    static std::vector<int> xywh_to_xyxy(int x, int y, int w, int h) {
        double w_2 = w / 2.0;
        double h_2 = h / 2.0;
        int x1     = (int)std::ceil(x - w_2);
        int x2     = (int)std::ceil(x + w_2);
        int y1     = (int)std::ceil(y - h_2);
        int y2     = (int)std::ceil(y + h_2);
        return std::vector<int>{x1, y1, x2, y2};
    }
    static std::vector<int> xywh_to_xyxy(std::vector<int> xywh) {
        double w_2 = xywh[2] / 2.0;
        double h_2 = xywh[3] / 2.0;
        int x1     = (int)std::ceil(xywh[0] - w_2);
        int x2     = (int)std::ceil(xywh[0] + w_2);
        int y1     = (int)std::ceil(xywh[1] - h_2);
        int y2     = (int)std::ceil(xywh[1] + h_2);
        return std::vector<int>{x1, y1, x2, y2};
    }
    static cv::Mat xywh_to_xyxy(cv::Mat xywh){
        cv::Mat xyxy(xywh.size(), CV_64F);
        cv::Mat half(xywh.rows, 1, CV_64F, cv::Scalar(0.5));
        cv::Mat w_2 = xywh.col(2).mul(half);
        cv::Mat h_2 = xywh.col(3).mul(half);
        cv::Mat mat_arr[] = {xywh.col(0) - w_2,
                             xywh.col(1) - h_2,
                             xywh.col(2) + w_2,
                             xywh.col(3) + h_2};
        cv::hconcat(mat_arr, 4, xyxy);

        cv::Mat xyxy_int(xywh.size(), CV_32S); // convert to int
        auto src_it = xyxy.begin<double>();
        auto dst_it = xyxy_int.begin<int>();
        for (; src_it != xyxy.end<double>; src_it ++, dst_it ++){
            (*dst_it) = (int)std::floor(*src_it);
        }
        return xyxy_int;
    }
};
}