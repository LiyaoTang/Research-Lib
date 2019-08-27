#include "anchor.hpp"
#include <stdexcept>

namespace utils {

Anchor::Anchor(int stride, std::vector<float> ratios, std::vector<float> scales,
               std::vector<float> iou_thr int pos_num, int neg_num, int total_num,
               std::string channel_order) : _stride(stride),
                                            _ratios(ratios),
                                            _scales(scales),
                                            _anchor_num(ratios.size() * scales.size()),
                                            _pos_num(pos_num),
                                            _neg_num(neg_num),
                                            _total_num(total_num) {
    _channel_order = channel_order.substr(channel_order.size() - 3, string::npos);
    if (_channel_order == "CHW") {
        _decompose_box = [](cv::Mat b) { return std::vector<cv::Mat>{b.row(0), b.row(1), b.row(2), b.row(3)}; }
    } else if (_channel_order == "HWC") {
        _decompose_box = [](cv::Mat b) { cv::Mat chw_box; cv::split(b, chw_box) return std::vector<cv::Mat>{b.row(0), b.row(1), b.row(2), b.row(3)}; }
    } else {
        throw std::invalid_argument("only NCHW/NHWC allowed (got \"" + _channel_order + "\" from \"" + channel_order + "\")");
    }

    _iou_thr = std::sort(iou_thr.begin(), iou_thr.end());
    if (_iou_thr.size() != 2) {
        std::ostringstream oss;
        std::copy(_iou_thr.begin(), _iou_thr.end() - 1, std::ostream_iterator<float>(oss, "-"));  // avoid trailing "-"
        oss << _iou_thr.back();
        throw std::invalid_argument("not implemented for iou threshold " + oss.str());
    }
}

cv::Mat Anchor::prepare_anchors() {
    cv::Mat()
}

}