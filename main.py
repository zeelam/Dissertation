import math
import rospy
import airsim
import numpy as np
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud

def pub_pointcloud(points):
    pc = PointCloud()
    pc.header.stamp = rospy.Time.now()
    pc.header.frame_id = 'lidar'

    for i in range(len(points)):
        pc.points.append(Point32(-points[i][0], points[i][1], -points[i][2]))
    # print("pc:", pc)
    return pc

def main():
    client = airsim.CarClient()
    client.confirmConnection()

    publisher = rospy.Publisher('/pointcloud', PointCloud, queue_size=10)
    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        lidarData = client.getLidarData()
        if len(lidarData.point_cloud) > 3:
            print(len(lidarData.point_cloud) / 3)
            points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            pc = pub_pointcloud(points)
            publisher.publish(pc)
            rate.sleep()
        else:
            print("No points received from Lidar")

if __name__ == '__main__':
    rospy.init_node('car_lidar', anonymous=True)
    main()