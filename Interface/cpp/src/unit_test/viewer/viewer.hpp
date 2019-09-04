#pragma once
#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace utils {

class Viewer {
public:
    Viewer() {}

    void set_image(const cv::Mat &image);

    void draw_bbox(const cv::Rect &bbox, const cv::Scalar &color = cv::Scalar(255, 255, 255));

    void draw_text(const std::string &text, const cv::Point &pt, const cv::Scalar &color = cv::Scalar(255, 255, 255));

    void show(const std::string &win_name="", int delay=0, float scale=1) const;

private:
    cv::Mat _image;
};

}  // namespace utils
