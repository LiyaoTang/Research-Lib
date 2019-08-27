#include <torch/script.h>
#include <iostream>
#include <opencv2/core.hpp>

namespace models {
namespace torch_model {

/** 
 * base class for torch tracker
 * **/
class Tracker {
public:
    Tracker(std::string backbone_path, std::string rpn_path, std::string bbox_encoding, int input_size=-1);
    void track_init(cv::Mat img, cv::Rect box, int track_id);
    void track(std::vector<cv::Mat>& img_list, cv::Rect& box, int track_id);
    cv::Rect track_onestep(cv::Mat img, int track_id);
    void track_fix(cv::Mat img, cv::Rect box, int track_id);

private:
    const int _input_size = -1; // model input size
    torch::jit::script::Module _extractor; // backbone & extraction
    torch::jit::script::Module _rpn; // output head

    std::map<int, std::pair<std::shared_ptr<torch::Tensor>, cv::Rect>> _track_pool;

    cv::Mat (Tracker::*_encode_bbox)(cv::Mat img, cv::Rect prev_box) = nullptr;
    cv::Mat _encode_bbox_crop(cv::Mat img, cv::Rect prev_box);
    cv::Mat _encode_bbox_mask(cv::Mat img, cv::Rect prev_box);

    cv::Mat (Tracker::*_decode_bbox)(cv::Mat img, cv::Rect pred_box) = nullptr;
    cv::Mat _decode_bbox_crop(cv::Mat img, cv::Rect pred_box);
    cv::Mat _decode_bbox_mask(cv::Mat img, cv::Rect pred_box);

    Tracker() {}

};

}  // namespace torch_model
}  // namespace models