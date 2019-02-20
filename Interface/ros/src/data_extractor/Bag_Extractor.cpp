
for (auto bag_file : bags_name){
        

        std::string bag_name = bags_dir + "/" + bag_file;
        try {
            _bag.open(bag_name, rosbag::bagmode::Read);
            std::cout << "Opening Bag:" << bag_name << std::endl;
        }
        catch (...){
            _bag.close();
            std::cout << " Failed to open Bag:" << bag_name << std::endl;
            continue;
        }

        rosbag::View view(_bag, rosbag::TopicQuery(_topics));
        BOOST_FOREACH (rosbag::MessageInstance const m ,view) {
            std::string topic_name = m.getTopic();
            perception_msgs::radar_targets::ConstPtr radar_targets_msg_ptr =
                                              m.instantiate<perception_msgs::radar_targets>();
            if (radar_targets_msg_ptr != nullptr) {
                if (topic_name == "/radar/front_targets") {
                    get_radar_targets(radar_targets_msg_ptr, front_radar_targets, 0);
                    **.pushback();
                }
                if (topic_name == "/radar/left_front_targets") {
                    get_radar_targets(radar_targets_msg_ptr, left_front_radar_targets, 0);
                    flag_left_front = 1;
                }
                if (topic_name == "/radar/right_front_targets") {
                    get_radar_targets(radar_targets_msg_ptr, right_front_radar_targets, 1);
                    flag_right_front = 1;
                }
                if (topic_name == "/radar/back_targets") {
                    get_radar_targets(radar_targets_msg_ptr, rear_radar_targets, 0);
                }
                if (topic_name == "/radar/left_back_targets") {
                    get_radar_targets(radar_targets_msg_ptr, left_rear_radar_targets, 1);
                    flag_left_rear = 1;
                }
                if (topic_name == "/radar/right_back_targets") {
                    get_radar_targets(radar_targets_msg_ptr, right_rear_radar_targets, 0);
                    flag_right_rear = 1;
                }
            }//end if (radar_targets_msg_ptr != nullptr)

            perception_msgs::vehicle_speed::ConstPtr vehicle_speed_msg_ptr =
                                              m.instantiate<perception_msgs::vehicle_speed>();
            if (vehicle_speed_msg_ptr != nullptr){
                vehicle_speed = vehicle_speed_msg_ptr->vehicle_speed;
            }

            sensor_msgs::CompressedImage::ConstPtr camera_msg_ptr =
                                           m.instantiate<sensor_msgs::CompressedImage>();
            if (camera_msg_ptr != nullptr) {
                cv::Mat front_compressed_image;
                cv::resize(cv::imdecode(cv::Mat(camera_msg_ptr->data), 1) ,front_compressed_image, cv::Size(720, 405), (0, 0), (0, 0), cv::INTER_LINEAR);
                cv::cvtColor(front_compressed_image, front_compressed_image, cv::COLOR_BGR2RGB, 0);
                cv::namedWindow("Video",CV_WINDOW_NORMAL);
                cv::imshow("Video",front_compressed_image);
                cv::waitKey(1);
            }
            /****************************/ 
  
            /****************************/

        } // end foreach (rosbag::MessageInstance const m, view)
        // add processing code here

        _bag.close();
    }

    std::vector<RadarInput> front_radar_targets;
    std::vector<RadarInput> left_front_radar_targets;
    std::vector<RadarInput> right_front_radar_targets;
    std::vector<RadarInput> rear_radar_targets;
    std::vector<RadarInput> left_rear_radar_targets;
    std::vector<RadarInput> right_rear_radar_targets;
    double vehicle_speed = 0;

    rosbag::Bag _bag;
    std::vector<std::string> _topics;

void CRosbagReader::set_topics()
{
    _topics.clear();
    _topics.push_back("/radar/front_targets");
    _topics.push_back("/radar/left_front_targets");
    _topics.push_back("/radar/right_front_targets");
    _topics.push_back("/radar/back_targets");
    _topics.push_back("/radar/left_back_targets");
    _topics.push_back("/radar/right_back_targets");
    _topics.push_back("/vehicle_speed");
    _topics.push_back("/camera/front_middle/compressed");
}


    void get_radar_targets(const perception_msgs::radar_targets::ConstPtr& msg,\
                           std::vector<RadarInput>& radar_targets,\
                           int flag);

void CRosbagReader::get_radar_targets(const perception_msgs::radar_targets::ConstPtr& msg, \
                                      std::vector<RadarInput>& radar_targets,\
                                      int flag)
{
    radar_targets.clear();
    float sym_flag = 1.0;
    switch (flag) {
    case 0:
        sym_flag = 1.0;
        break;
    case 1:
        sym_flag = -1.0;
        break;
    default:
        break;
    }
    if (msg->radar_targets.size() > 0) {
        for(int i = 0; i < msg->radar_targets.size(); i++){
            RadarInput tmp_radar_target;
            tmp_radar_target.timestamp = msg->radar_targets[i].timestamp;
            tmp_radar_target.group_id = msg->radar_targets[i].group_id;
            tmp_radar_target.target_id = msg->radar_targets[i].target_id;
            tmp_radar_target.track_status = msg->radar_targets[i].track_status;
            tmp_radar_target.distance =
                    sqrt(msg->radar_targets[i].coordinate.x * msg->radar_targets[i].coordinate.x
                         + msg->radar_targets[i].coordinate.y * msg->radar_targets[i].coordinate.y);

            tmp_radar_target.coordinate.x = msg->radar_targets[i].coordinate.x;
            tmp_radar_target.coordinate.y = sym_flag * msg->radar_targets[i].coordinate.y;
            tmp_radar_target.angle = \
                    atan2(tmp_radar_target.coordinate.y, tmp_radar_target.coordinate.x);

            tmp_radar_target.velocity.x = tmp_radar_target.coordinate.x;
            tmp_radar_target.velocity.y = tmp_radar_target.coordinate.y;
            tmp_radar_target.speed = msg->radar_targets[i].velocity.z;
            tmp_radar_target.rcs = msg->radar_targets[i].rcs;
            tmp_radar_target.target_type = ObjectType(msg->radar_targets[i].obj_type);

            radar_targets.push_back(tmp_radar_target);
        }
    }

}



