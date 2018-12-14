#include "ros/ros.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Fuck this shit";


class ImageConverter {
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Publisher image_pub_;
  
  public:
    ImageConverter() :it_(nh_) {
      image_sub_ = it_.subscribe("/kinect2/sd/image_ir_rect", 1, &ImageConverter::imageCb, this);
      image_pub_ = it_.advertise("/image_converter/output_stream", 1);

      cv::namedWindow(OPENCV_WINDOW);
    }
    ~ImageConverter() {
      cv::destroyWindow(OPENCV_WINDOW);
    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg) {
      cv_bridge::CvImagePtr cv_ptr;
      try {
        cv_bridge::CvtColorForDisplayOptions options;
        options.do_dynamic_scaling = true;
        //options.
        options.min_image_value = 0
        options.max_image_value = 10*100;
        cv_ptr = cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(msg), "", options);
      } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
      cv::imshow(OPENCV_WINDOW, cv_ptr->image);
      cv::waitKey(3);

      image_pub_.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "image_converter");
  ImageConverter ic;
  ros::spin();
  return 0;
}
