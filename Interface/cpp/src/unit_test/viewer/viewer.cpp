#include "viewer.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace utils {

void Viewer::set_image(const cv::Mat &image) {
    _image = image.clone();
}

void Viewer::draw_bbox(const cv::Rect &bbox, const cv::Scalar &color) {
    cv::rectangle(_image, bbox, color, 2);
}

void Viewer::draw_text(const std::string &text, const cv::Point &pt, const cv::Scalar &color) {
    cv::putText(_image, text, pt, cv::FONT_HERSHEY_COMPLEX, 2, color, 2);
}

void Viewer::show(const std::string &win_name, int delay, float scale) const {
    cv::Mat img;
    cv::resize(_image, img, cv::Size(int(_image.cols * scale + 0.5),
                                     int(_image.rows * scale + 0.5)));
    cv::imshow(win_name, img);
    cv::waitKey(delay);
}

}  // namespace utils