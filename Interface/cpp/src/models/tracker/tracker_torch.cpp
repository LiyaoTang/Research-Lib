#include "tracker_torch.hpp"
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace models {
namespace torch_model {

/** 
 * implementation: base class of torch tracker
 * **/
Tracker::Tracker(std::string model_path, std::string bbox_encoding, int input_size) {
    if (bbox_encoding == "crop") {
        _encode_bbox = std::bind(&Tracker::_encode_bbox_crop, this, std::placeholders::_1, std::placeholders::_2);
        // _decode_bbox = std::bind(&Tracker::_decode_bbox_crop, this, std::placeholders::_1, std::placeholders::_2);
    } else if (bbox_encoding == "mask") {
        _encode_bbox = std::bind(&Tracker::_encode_bbox_mask, this, std::placeholders::_1, std::placeholders::_2);
        // _decode_bbox = std::bind(&Tracker::_decode_bbox_mask, this, std::placeholders::_1, std::placeholders::_2);
    } else {
        throw;
    }
    _model = torch::jit::load(model_path);
}

void Tracker::track_init(cv::Mat img, cv::Rect box, int track_id) {
    cv::Mat input = _encode_bbox(img, box);
    // in-place construction & assume 8-bit img (by specifying each element as Byte)
    torch::Tensor input_tensor = torch::from_blob(input.data, {1, input.channels(), input.rows, input.cols}, torch::kByte);
    std::vector<torch::IValue> model_input{input_tensor.to(torch::kFloat)};  // convert to accepted input
    torch::Tensor model_out = _model.forward(model_input).toTensor();

    /**
     * torch::IValue input = 
     * _model.run_method("init", )
     * **/

    _track_pool[track_id].first  = std::make_shared<torch::Tensor>(model_out);
    _track_pool[track_id].second = box;
}

cv::Mat Tracker::_encode_bbox_crop(cv::Mat img, cv::Rect prev_box) {  // siamRPN crop
    double context = (prev_box.width + prev_box.height + 0.0) / 2;
    double sz      = std::sqrt((prev_box.width + context) * (prev_box.height + context));
    int x          = prev_box.x + prev_box.width / 2 - int(sz / 2);
    int y          = prev_box.y + prev_box.height / 2 - int(sz / 2);
    cv::Rect crop_region(x, y, (int)sz, (int)sz);  // desired region
    cv::Mat board((int)sz, (int)sz, CV_8U);
    if (x < 0 || y < 0 || x + (int)sz > img.cols || y + (int)sz > img.rows) {
        // calc offset of actual crop within desired crop & pad for exceeded region
        board.setTo(cv::mean(img));
        cv::Rect valid_area(x, y, (int)sz, (int)sz);  // actual region
        if (x < 0) valid_area.x = -x;
        if (y < 0) valid_area.y = -y;
        if (x + (int)sz > img.cols) valid_area.width -= x + (int)sz - img.cols;
        if (y + (int)sz > img.rows) valid_area.height -= y + (int)sz - img.rows;
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
    auto tracked_data = _track_pool[track_id];
    cv::Rect box      = tracked_data.second;
    auto zf           = tracked_data.first;

    // extract on search region
    cv::Mat x_crop         = _encode_bbox(img, box);
    torch::Tensor x_crop_t = torch::from_blob(x_crop.data, {1, x_crop.channels(), x_crop.rows, x_crop.cols}, torch::kByte);
    // torch::IValue out      = _model.run_method("track", zf, x_crop_t);
    // auto out_list = out.toTensorList();
    // _track_pool[track_id].second = out_list[1];
    // _track_pool[track_id].second = out_list[0];
}

void Tracker::track_fix(cv::Mat img, cv::Rect box, int track_id) {
    track_init(img, box, track_id);
}

}  // namespace torch_model
}  // namespace models