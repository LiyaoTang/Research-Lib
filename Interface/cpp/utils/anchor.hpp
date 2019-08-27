#pragma once
#include <functional>
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "bbox.hpp"

namespace utils {

class Anchor {
public:
    Anchor(int stride, std::vector<double> ratios, std::vector<double> scales,
           std::vector<double> iou_thr = std::vector<double>{0.3, 0.6},
           int pos_num = 16, int neg_num = 16, int total_num = 64, std::string channel_order = "CHW");

    cv::Mat decode_bbox(cv::Mat pred, std::pair<double, double> img_center, size_t size);

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

    cv::Mat _anchors;
    cv::Mat _anchors_volume;
    std::function<std::vector<cv::Mat>(cv::Mat)> _decompose_box = NULL;

private:
    cv::Mat prepare_anchors();
    cv::Mat generate_anchors_volume(int size, std::pair<int, int> img_center=std::pair<int, int>(0,0));
};

// template <typename Dtype>
// void ProposalLayer<Dtype>::Generate_anchors() {
//     vector<double> base_anchor;
//     base_anchor.push_back(0);
//     base_anchor.push_back(0);
//     base_anchor.push_back(anchor_base_size_);
//     base_anchor.push_back(anchor_base_size_);
//     vector<double> anchors_ratio;
//     _ratio_enum(base_anchor, anchors_ratio);
//     _scale_enum(anchors_ratio, anchor_boxes_);
// }

// template <typename Dtype>
// void ProposalLayer<Dtype>::_whctrs(vector<double> anchor, vector<double> &ctrs) {
//     double w     = anchor[2] - anchor[0] + 1;
//     double h     = anchor[3] - anchor[1] + 1;
//     double x_ctr = anchor[0] + 0.5 * (w - 1);
//     double y_ctr = anchor[1] + 0.5 * (h - 1);
//     ctrs.push_back(w);
//     ctrs.push_back(h);
//     ctrs.push_back(x_ctr);
//     ctrs.push_back(y_ctr);
// }

// template <typename Dtype>
// void ProposalLayer<Dtype>::_ratio_enum(vector<double> anchor, vector<double> &anchors_ratio) {
//     vector<double> ctrs;
//     _whctrs(anchor, ctrs);
//     double size    = ctrs[0] * ctrs[1];
//     int ratio_num = anchor_ratio_.size();
//     for (int i = 0; i < ratio_num; i++) {
//         double ratio = size / anchor_ratio_[i];
//         int ws      = int(round(sqrt(ratio)));
//         int hs      = int(round(ws * anchor_ratio_[i]));
//         vector<double> ctrs_in;
//         ctrs_in.push_back(ws);
//         ctrs_in.push_back(hs);
//         ctrs_in.push_back(ctrs[2]);
//         ctrs_in.push_back(ctrs[3]);
//         _mkanchors(ctrs_in, anchors_ratio);
//     }
// }

// template <typename Dtype>
// void ProposalLayer<Dtype>::_scale_enum(vector<double> anchors_ratio, vector<double> &anchor_boxes) {
//     int anchors_ratio_num = anchors_ratio.size() / 4;
//     for (int i = 0; i < anchors_ratio_num; i++) {
//         vector<double> anchor;
//         anchor.push_back(anchors_ratio[i * 4]);
//         anchor.push_back(anchors_ratio[i * 4 + 1]);
//         anchor.push_back(anchors_ratio[i * 4 + 2]);
//         anchor.push_back(anchors_ratio[i * 4 + 3]);
//         vector<double> ctrs;
//         _whctrs(anchor, ctrs);
//         int scale_num =
}