// Generated by gencpp from file perception_msgs/radar_object.msg
// DO NOT EDIT!


#ifndef PERCEPTION_MSGS_MESSAGE_RADAR_OBJECT_H
#define PERCEPTION_MSGS_MESSAGE_RADAR_OBJECT_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <perception_msgs/coord3d.h>
#include <perception_msgs/coord3d.h>
#include <perception_msgs/coord3d.h>
#include <perception_msgs/coord3d.h>
#include <perception_msgs/track_flag.h>

namespace perception_msgs
{
template <class ContainerAllocator>
struct radar_object_
{
  typedef radar_object_<ContainerAllocator> Type;

  radar_object_()
    : header()
    , timestamp(0)
    , group_id(0)
    , track_id(0)
    , track_status(0)
    , obj_type(0)
    , coordinate()
    , velocity()
    , acceleration()
    , size_of_box()
    , rcs(0.0)
    , background_level(0)
    , prob(0.0)
    , track_flag()  {
    }
  radar_object_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , timestamp(0)
    , group_id(0)
    , track_id(0)
    , track_status(0)
    , obj_type(0)
    , coordinate(_alloc)
    , velocity(_alloc)
    , acceleration(_alloc)
    , size_of_box(_alloc)
    , rcs(0.0)
    , background_level(0)
    , prob(0.0)
    , track_flag(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef uint64_t _timestamp_type;
  _timestamp_type timestamp;

   typedef uint32_t _group_id_type;
  _group_id_type group_id;

   typedef uint32_t _track_id_type;
  _track_id_type track_id;

   typedef uint32_t _track_status_type;
  _track_status_type track_status;

   typedef uint32_t _obj_type_type;
  _obj_type_type obj_type;

   typedef  ::perception_msgs::coord3d_<ContainerAllocator>  _coordinate_type;
  _coordinate_type coordinate;

   typedef  ::perception_msgs::coord3d_<ContainerAllocator>  _velocity_type;
  _velocity_type velocity;

   typedef  ::perception_msgs::coord3d_<ContainerAllocator>  _acceleration_type;
  _acceleration_type acceleration;

   typedef  ::perception_msgs::coord3d_<ContainerAllocator>  _size_of_box_type;
  _size_of_box_type size_of_box;

   typedef double _rcs_type;
  _rcs_type rcs;

   typedef uint32_t _background_level_type;
  _background_level_type background_level;

   typedef double _prob_type;
  _prob_type prob;

   typedef  ::perception_msgs::track_flag_<ContainerAllocator>  _track_flag_type;
  _track_flag_type track_flag;





  typedef boost::shared_ptr< ::perception_msgs::radar_object_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::perception_msgs::radar_object_<ContainerAllocator> const> ConstPtr;

}; // struct radar_object_

typedef ::perception_msgs::radar_object_<std::allocator<void> > radar_object;

typedef boost::shared_ptr< ::perception_msgs::radar_object > radar_objectPtr;
typedef boost::shared_ptr< ::perception_msgs::radar_object const> radar_objectConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::perception_msgs::radar_object_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::perception_msgs::radar_object_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace perception_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': True}
// {'perception_msgs': ['/home/shawn/baidu/ai-auto/l3-apollo/catkin_build/src/modules/ros_msgs/perception/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::perception_msgs::radar_object_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::perception_msgs::radar_object_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::perception_msgs::radar_object_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::perception_msgs::radar_object_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::perception_msgs::radar_object_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::perception_msgs::radar_object_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::perception_msgs::radar_object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "be3228555d03c4e2339d2fd37d74a2bc";
  }

  static const char* value(const ::perception_msgs::radar_object_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xbe3228555d03c4e2ULL;
  static const uint64_t static_value2 = 0x339d2fd37d74a2bcULL;
};

template<class ContainerAllocator>
struct DataType< ::perception_msgs::radar_object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "perception_msgs/radar_object";
  }

  static const char* value(const ::perception_msgs::radar_object_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::perception_msgs::radar_object_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n\
uint64 timestamp    # unit: us\n\
uint32 group_id     # radar id (front: 0; left front: 1; right front: 2)\n\
uint32 track_id\n\
uint32 track_status\n\
uint32 obj_type\n\
coord3d coordinate\n\
coord3d velocity\n\
coord3d acceleration\n\
coord3d size_of_box\n\
float64 rcs\n\
uint32 background_level\n\
float64 prob\n\
track_flag track_flag\n\
\n\
================================================================================\n\
MSG: std_msgs/Header\n\
# Standard metadata for higher-level stamped data types.\n\
# This is generally used to communicate timestamped data \n\
# in a particular coordinate frame.\n\
# \n\
# sequence ID: consecutively increasing ID \n\
uint32 seq\n\
#Two-integer timestamp that is expressed as:\n\
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n\
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n\
# time-handling sugar is provided by the client library\n\
time stamp\n\
#Frame this data is associated with\n\
# 0: no frame\n\
# 1: global frame\n\
string frame_id\n\
\n\
================================================================================\n\
MSG: perception_msgs/coord3d\n\
float64 x\n\
float64 y\n\
float64 z\n\
\n\
================================================================================\n\
MSG: perception_msgs/track_flag\n\
Header header\n\
int32 find_times\n\
int32 wrong_times\n\
int32 lost_times\n\
int32 threshold_find\n\
int32 threshold_wrong\n\
int32 threshold_lost\n\
";
  }

  static const char* value(const ::perception_msgs::radar_object_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::perception_msgs::radar_object_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.timestamp);
      stream.next(m.group_id);
      stream.next(m.track_id);
      stream.next(m.track_status);
      stream.next(m.obj_type);
      stream.next(m.coordinate);
      stream.next(m.velocity);
      stream.next(m.acceleration);
      stream.next(m.size_of_box);
      stream.next(m.rcs);
      stream.next(m.background_level);
      stream.next(m.prob);
      stream.next(m.track_flag);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct radar_object_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::perception_msgs::radar_object_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::perception_msgs::radar_object_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "timestamp: ";
    Printer<uint64_t>::stream(s, indent + "  ", v.timestamp);
    s << indent << "group_id: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.group_id);
    s << indent << "track_id: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.track_id);
    s << indent << "track_status: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.track_status);
    s << indent << "obj_type: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.obj_type);
    s << indent << "coordinate: ";
    s << std::endl;
    Printer< ::perception_msgs::coord3d_<ContainerAllocator> >::stream(s, indent + "  ", v.coordinate);
    s << indent << "velocity: ";
    s << std::endl;
    Printer< ::perception_msgs::coord3d_<ContainerAllocator> >::stream(s, indent + "  ", v.velocity);
    s << indent << "acceleration: ";
    s << std::endl;
    Printer< ::perception_msgs::coord3d_<ContainerAllocator> >::stream(s, indent + "  ", v.acceleration);
    s << indent << "size_of_box: ";
    s << std::endl;
    Printer< ::perception_msgs::coord3d_<ContainerAllocator> >::stream(s, indent + "  ", v.size_of_box);
    s << indent << "rcs: ";
    Printer<double>::stream(s, indent + "  ", v.rcs);
    s << indent << "background_level: ";
    Printer<uint32_t>::stream(s, indent + "  ", v.background_level);
    s << indent << "prob: ";
    Printer<double>::stream(s, indent + "  ", v.prob);
    s << indent << "track_flag: ";
    s << std::endl;
    Printer< ::perception_msgs::track_flag_<ContainerAllocator> >::stream(s, indent + "  ", v.track_flag);
  }
};

} // namespace message_operations
} // namespace ros

#endif // PERCEPTION_MSGS_MESSAGE_RADAR_OBJECT_H