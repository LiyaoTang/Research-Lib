#include "anchor.hpp"
#include <iterator>
#include <stdexcept>

namespace utils {

Anchor::Anchor(int stride, std::vector<double> ratios, std::vector<double> scales,
               std::vector<double> iou_thr, int pos_num, int neg_num, int total_num,
               std::string channel_order) : _stride(stride),
                                            _ratios(ratios),
                                            _scales(scales),
                                            _anchor_num(ratios.size() * scales.size()),
                                            _pos_num(pos_num),
                                            _neg_num(neg_num),
                                            _total_num(total_num),
                                            _img_center(std::pair<int, int>(0, 0)),
                                            _size(0) {
    _channel_order = channel_order.substr(channel_order.size() - 3, std::string::npos);
    if (_channel_order == "CHW") {
        _decompose_box = [](cv::Mat b) { return std::vector<cv::Mat>{b.row(0), b.row(1), b.row(2), b.row(3)}; };
    } else if (_channel_order == "HWC") {
        _decompose_box = [](cv::Mat b) { cv::Mat chw_box; cv::split(b, chw_box); return std::vector<cv::Mat>{b.row(0), b.row(1), b.row(2), b.row(3)}; };
    } else {
        throw std::invalid_argument("only NCHW/NHWC allowed (got \"" + _channel_order + "\" from \"" + channel_order + "\")");
    }

    std::sort(iou_thr.begin(), iou_thr.end());
    if (_iou_thr.size() != 2) {
        std::ostringstream oss;
        std::copy(_iou_thr.begin(), _iou_thr.end() - 1, std::ostream_iterator<double>(oss, "-"));  // avoid trailing "-"
        oss << _iou_thr.back();
        throw std::invalid_argument("not implemented for iou threshold " + oss.str());
    }

    prepare_anchors();  // prepare anchor for one location
}

cv::Mat Anchor::prepare_anchors() {
    _anchors = cv::Mat::zeros(cv::Size(_anchor_num, 4), _cv_type);
    int cnt  = 0;
    for (int i = 0; i < _ratios.size(); i++) {
        double sqrt_r = std::sqrt(_ratios[i]);
        int ws        = int(_stride / sqrt_r);
        int hs        = int(_stride * sqrt_r);

        for (int j = 0; j < _scales.size(); j++) {
            int w = ws * _scales[j];
            int h = hs * _scales[j];
            std::memcpy(_anchors.ptr<int>(cnt), std::vector<int>{0, 0, w, h}.data, _anchors.cols * sizeof(int));
        }
    }
    return _anchors;  // xywh
}

cv::Mat Anchor::generate_anchors_volume(int size, std::pair<int, int> img_center) {
    if (_img_center == img_center && _size == size) {
        return _anchors_volume;
    }
    _img_center = img_center;
    _size       = size;

    cv::Mat anchor = cv::repeat(_anchors, size * size, 1);  // [anchor_num*size*size, 4]
    std::pair<int, int> ori = {img_center.first - size / 2 * _stride,
                               img_center.second - size / 2 * _stride};
    // get mesh grid: xx, yy [size*size*anchor_num]
    std::vector<int> range_x;
    std::vector<int> range_y;
    for (int i = 0; i < size; i++) {
        range_x.push_back(i * _stride + ori.first);
        range_y.push_back(i * _stride + ori.second);
    }
    cv::Mat xx(1, range_x.size(), _cv_type, range_x.data);
    xx = cv::repeat(xx.t(), size, 1);
    xx = cv::repeat(xx, _anchor_num, 1);
    xx = xx.reshape(0, 1);

    cv::Mat yy(1, range_y.size(), _cv_type, range_y.data);
    yy = cv::repeat(yy, 1, size);
    yy = cv::repeat(yy, _anchor_num, 1);
    yy = yy.reshape(0, 1);

    // copy cx-cy
    xx.copyTo(anchor.col(0));
    yy.copyTo(anchor.col(1));

    _anchors_volume = anchor;
    return _anchors_volume;
}
}