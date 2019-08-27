#include "tracker_torch.hpp"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace models {
namespace torch_model {

/** 
 * implementation: base class of torch tracker
 * **/
Tracker::Tracker(std::string backbone_path, std::string rpn_path, std::string bbox_encoding, int input_size) : _input_size(input_size) {
    if (bbox_encoding == "crop") {
        _encode_bbox = _encode_bbox_crop;
        _decode_bbox = _decode_bbox_crop;
    } else if (bbox_encoding == "mask") {
        _encode_bbox = _encode_bbox_mask;
        _decode_bbox = _decode_bbox_mask;
    } else {
        throw;
    }
    _extractor = torch::jit::load(backbone_path);
    _rpn       = torch::jit::load(rpn_path);
}

void Tracker::track_init(cv::Mat img, cv::Rect box, int track_id) {
    cv::Mat input = _encode_bbox(img, box);
    // in-place construction & assume 8-bit img (by specifying each element as Byte)
    torch::Tensor input_tensor = torch::from_blob(input.data, {1, input.channels(), input.rows, input.cols}, torch::kByte);
    std::vector<torch::IValue> model_input{input_tensor.to(torch::kFloat)};  // convert to accepted input
    torch::Tensor model_out = _extractor.forward(model_input).toTensor();

    _track_pool[track_id].first  = std::make_shared<torch::Tensor>(model_out);
    _track_pool[track_id].second = box;
}

cv::Mat Tracker::_encode_bbox_crop(cv::Mat img, cv::Rect prev_box) {  // siamRPN crop
    double context = (prev_box.width + prev_box.height + 0.0) / 2;
    double sz      = std::sqrt((prev_box.width + context) * (prev_box.height + context));
    int x          = prev_box.x + prev_box.width / 2 - int(sz / 2);
    int y          = prev_box.y + prev_box.height / 2 - int(sz / 2);
    cv::Rect crop_region(x, y, int(sz), int(sz));  // desired region
    cv::Mat board(int(sz), int(sz), CV_8U);
    if (x < 0 || y < 0 || x + int(sz) > img.cols || y + int(sz) > img.rows) {
        // calc offset of actual crop within desired crop & pad for exceeded region
        board.setTo(cv::mean(img));
        cv::Rect valid_area(x, y, int(sz), int(sz));  // actual region
        if (x < 0) valid_area.x = -x;
        if (y < 0) valid_area.y = -y;
        if (x + int(sz) > img.cols) valid_area.width -= x + int(sz) - img.cols;
        if (y + int(sz) > img.rows) valid_area.height -= y + int(sz) - img.rows;
        img(crop_region).copyTo(board(valid_area));
    } else {
        img(crop_region).copyTo(board);
    }
    cv::Mat resized_crop;
    cv::resize(board, resized_crop, cv::Size(_input_size, _input_size));
    return resized_crop;
}

cv::Mat Tracker::_encode_bbox_mask(cv::Mat img, cv::Rect prev_box) {
    cv::Mat img_copy;
    img.copyTo(img_copy);

    std::vector<cv::Mat> ch_vec;
    cv::split(img_copy, ch_vec);
    cv::Mat mask(img.rows, img.cols, CV_32F, cv::Scalar(0));
    mask(prev_box).setTo(1);
    ch_vec.push_back(mask);

    cv::Mat merged;
    cv::merge(ch_vec, merged);
    return merged;
}

void Tracker::track(std::vector<cv::Mat>& img_list, cv::Rect& box, int track_id) {
}

cv::Rect Tracker::track_onestep(cv::Mat img, int track_id) {
    cv::Rect box = _track_pool[track_id].second;

    // extract on search region
    cv::Mat input              = _encode_bbox(img, box);
    torch::Tensor input_tensor = torch::from_blob(input.data, {1, input.channels(), input.rows, input.cols}, torch::kByte);
    std::vector<torch::IValue> model_input{input_tensor.to(torch::kFloat)};  // convert to accepted input
    torch::Tensor xf = _extractor.forward(model_input).toTensor();

    // rpn(xf, zf) -> (box, cls)
    std::vector<torch::IValue> rpn_input{xf, *(_track_pool[track_id].first)};
    torch::Tensor rpn_out = _rpn.forward(rpn_input).toTensor();
    rpn_out.slice(1, 0, 2);
    _track_pool[track_id].second;
}

void Tracker::track_fix(cv::Mat img, cv::Rect box, int track_id) {
    track_init(img, box, track_id);
}
};

}  // namespace torch_model
}  // namespace models