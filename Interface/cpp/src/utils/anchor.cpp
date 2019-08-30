#include "anchor.hpp"
#include <iterator>
#include <stdexcept>
#include <string>
#include "bbox.hpp"

namespace utils {

Anchor::Anchor(int stride, std::vector<double> ratios, std::vector<double> scales,
               std::vector<double> iou_thr, int pos_num, int neg_num, int total_num,
               std::string channel_order) : _stride(stride),
                                            _ratios(ratios),
                                            _scales(scales),
                                            _anchor_num(ratios.size() * scales.size()),
                                            _iou_thr(iou_thr),
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

    std::sort(_iou_thr.begin(), _iou_thr.end());
    if (_iou_thr.size() != 2) {
        std::ostringstream oss;
        std::copy(_iou_thr.begin(), _iou_thr.end() - 1, std::ostream_iterator<double>(oss, "-"));  // avoid trailing "-"
        oss << _iou_thr.back();
        throw std::invalid_argument("not implemented for iou threshold " + oss.str());
    }

    prepare_anchors();  // prepare anchor for one location
}

void Anchor::prepare_anchors() {
    cv::Mat cur_anchor = cv::Mat::zeros(1, 4, _cv_type);
    int cnt            = 0;
    for (int i = 0; i < _ratios.size(); i++) {
        double sqrt_r = std::sqrt(_ratios[i]);
        int ws        = int(_stride / sqrt_r);
        int hs        = int(_stride * sqrt_r);

        for (int j = 0; j < _scales.size(); j++) {
            int w = ws * _scales[j];
            int h = hs * _scales[j];

            cv::Mat anchor    = cv::Mat::zeros(1, 4, _cv_type);
            anchor.at<int>(2) = w;
            anchor.at<int>(3) = h;
            _anchors.push_back(anchor);
            cnt++;
        }
    }
}

cv::Mat Anchor::generate_anchors_volume(int size, std::pair<int, int> img_center) {
    if (_img_center == img_center && _size == size) {
        return _anchors_volume;
    }
    _img_center = img_center;
    _size       = size;

    // explicitly mimic numpy behavior: duplicate each row separately, instead of the whole arr
    cv::Mat anchor_vol_arr[_anchor_num];
    for (int i = 0; i < _anchor_num; i++) {
        anchor_vol_arr[i] = cv::repeat(_anchors[i], size * size, 1);
    }
    cv::Mat anchor_vol;
    cv::vconcat(anchor_vol_arr, _anchor_num, anchor_vol);  // [anchor_num*size*size, 4]

    std::pair<int, int> ori = {img_center.first - size / 2 * _stride,
                               img_center.second - size / 2 * _stride};
    // get mesh grid: xx, yy [size*size*anchor_num]
    cv::Mat xx(1, size, _cv_type);
    cv::Mat yy(1, size, _cv_type);
    for (int i = 0; i < size; i++) {
        xx.at<int>(i) = i * _stride + ori.second;  // col
        yy.at<int>(i) = i * _stride + ori.first;   // row
    }

    xx = cv::repeat(xx, size, 1);
    xx = cv::repeat(xx, _anchor_num, 1);
    xx = xx.reshape(0, size * size * _anchor_num);

    yy = cv::repeat(yy.t(), 1, size);
    yy = cv::repeat(yy, _anchor_num, 1);
    yy = yy.reshape(0, size * size * _anchor_num);

    // copy cx-cy
    xx.copyTo(anchor_vol.col(0));
    yy.copyTo(anchor_vol.col(1));

    _anchors_volume = anchor_vol;
    return _anchors_volume;
}

cv::Mat Anchor::decode_bbox(cv::Mat pred, std::pair<int, int> img_center, int size) {
    cv::Mat anchor_xywh           = generate_anchors_volume(size, img_center);
    cv::Mat anchor_xyxy           = bbox::xywh_to_xyxy(anchor_xywh);
    std::vector<cv::Mat> pred_box = _decompose_box(pred);  // p-xywh

    cv::Mat x = pred_box[0].mul(anchor_xywh.col(2)) + anchor_xywh.col(0);
    cv::Mat y = pred_box[1].mul(anchor_xywh.col(3)) + anchor_xywh.col(1);
    cv::Mat w, h;
    cv::exp(pred_box[2], w);
    cv::exp(pred_box[3], h);
    w = w.mul(anchor_xywh.col(2));
    h = h.mul(anchor_xywh.col(3));

    cv::Mat mat_arr[] = {x, y, w, h};
    cv::Mat rst;
    cv::hconcat(mat_arr, 4, rst);
    return rst;
}

}  // namespace utils