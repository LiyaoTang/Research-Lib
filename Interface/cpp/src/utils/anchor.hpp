#pragma once
#include <functional>
#include <iterator>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace utils {

class Anchor {
public:
    Anchor(int stride, std::vector<double> ratios, std::vector<double> scales,
           std::vector<double> iou_thr = {0.3, 0.6}, int pos_num = 16, int neg_num = 16,
           int total_num = 64, std::string channel_order = "CHW");

    cv::Mat generate_anchors_volume(int size, std::pair<int, int> img_center = {0, 0});
    cv::Mat decode_bbox(cv::Mat pred, std::pair<int, int> img_center, int size);

private:
    int _stride;
    std::vector<double> _ratios;
    std::vector<double> _scales;
    std::vector<double> _iou_thr;
    int _anchor_num;
    int _pos_num;
    int _neg_num;
    int _total_num;
    std::string _channel_order;
    const int _cv_type = CV_32S;

    std::pair<int, int> _img_center;
    int _size;

    std::vector<cv::Mat> _anchors;
    cv::Mat _anchors_volume;
    std::function<std::vector<cv::Mat>(cv::Mat)> _decompose_box = NULL;

private:
    void prepare_anchors();
};
}