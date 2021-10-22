import os
import math

import numpy as np
import airsim
import cv2
import time
from scipy.spatial.transform import Rotation as R
import argparse

class KITTI_data_format(object):
    def __init__(self):
        self.type = ''
        self.truncated = 0.0
        self.occluded = 0

        self.alpha = 0

        self.left = 0
        self.top = 0
        self.right = 0
        self.bottom = 0

        self.height = 0
        self.width = 0
        self.length = 0

        self.x = 0
        self.y = 0
        self.z = 0

        self.rotation_y = 0

    def __str__(self):
        return '{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}'.format(self.type, self.truncated, self.occluded,
                                                                     self.alpha, self.left, self.top, self.right,
                                                                     self.bottom, self.height, self.width, self.length,
                                                                     self.x, self.y, self.z, self.rotation_y)

class ViewRect(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    @classmethod
    def non_value(cls):
        return cls(0, 0, math.inf, math.inf)

def calculateProjectWorldToScreen(view_project_matrix, world_position, view_rect):
    world_position = np.insert(world_position, 3, 1.0)
    result = world_position.dot(view_project_matrix)
    # print(result)
    w = result[3]
    if w > 0:
        RHW = 1 / w
        result = np.delete(result, 3, 0)
        position_in_camera_space = result * RHW
        normalized_x = (position_in_camera_space[0] / 2) + 0.5
        normalized_y = 1 - (position_in_camera_space[1] / 2) - 0.5
        return np.array([normalized_x * view_rect.max_x - view_rect.min_x, normalized_y * view_rect.max_y - view_rect.min_y])
    else:
        return False

def draw2DBoundingBoxInImage(image, left, top, right, bottom):
    cv2.rectangle(image, (int(left), int(top)),
                  (int(right), int(bottom)), (0, 255, 0), 1)

def showLabelInImage(image, left, top, label):
        cv2.putText(image, label, (int(left), int(top - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12))

def draw3DBoundingBoxInImage(image, center_x, center_y, center_z, extend_x, extend_y, extend_z, rotation_mat,
                             view_projection_matrix, view_rect):
    extend_1 = np.array([extend_x, extend_y, extend_z]).dot(rotation_mat)
    extend_2 = np.array([-extend_x, extend_y, extend_z]).dot(rotation_mat)
    extend_3 = np.array([extend_x, -extend_y, extend_z]).dot(rotation_mat)
    extend_4 = np.array([-extend_x, -extend_y, extend_z]).dot(rotation_mat)
    extend_5 = np.array([extend_x, extend_y, -extend_z]).dot(rotation_mat)
    extend_6 = np.array([-extend_x, extend_y, -extend_z]).dot(rotation_mat)
    extend_7 = np.array([extend_x, -extend_y, -extend_z]).dot(rotation_mat)
    extend_8 = np.array([-extend_x, -extend_y, -extend_z]).dot(rotation_mat)

    # get 3D bounding box points in world space
    box3D_point_1 = np.array([center_x, center_y, center_z]) + extend_1
    box3D_point_2 = np.array([center_x, center_y, center_z]) + extend_2
    box3D_point_3 = np.array([center_x, center_y, center_z]) + extend_3
    box3D_point_4 = np.array([center_x, center_y, center_z]) + extend_4
    box3D_point_5 = np.array([center_x, center_y, center_z]) + extend_5
    box3D_point_6 = np.array([center_x, center_y, center_z]) + extend_6
    box3D_point_7 = np.array([center_x, center_y, center_z]) + extend_7
    box3D_point_8 = np.array([center_x, center_y, center_z]) + extend_8

    box3D_point_in_image_1 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_1, view_rect)
    box3D_point_in_image_2 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_2, view_rect)
    box3D_point_in_image_3 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_3, view_rect)
    box3D_point_in_image_4 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_4, view_rect)
    box3D_point_in_image_5 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_5, view_rect)
    box3D_point_in_image_6 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_6, view_rect)
    box3D_point_in_image_7 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_7, view_rect)
    box3D_point_in_image_8 = calculateProjectWorldToScreen(view_projection_matrix, box3D_point_8, view_rect)

    # draw 3D box in image
    cv2.line(image, tuple(np.around(box3D_point_in_image_1).astype(int)),
             tuple(np.around(box3D_point_in_image_2).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_2).astype(int)),
             tuple(np.around(box3D_point_in_image_4).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_4).astype(int)),
             tuple(np.around(box3D_point_in_image_3).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_3).astype(int)),
             tuple(np.around(box3D_point_in_image_1).astype(int)), (255, 0, 0), 1)

    cv2.line(image, tuple(np.around(box3D_point_in_image_5).astype(int)),
             tuple(np.around(box3D_point_in_image_6).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_6).astype(int)),
             tuple(np.around(box3D_point_in_image_8).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_8).astype(int)),
             tuple(np.around(box3D_point_in_image_7).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_7).astype(int)),
             tuple(np.around(box3D_point_in_image_5).astype(int)), (255, 0, 0), 1)

    cv2.line(image, tuple(np.around(box3D_point_in_image_1).astype(int)),
             tuple(np.around(box3D_point_in_image_5).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_2).astype(int)),
             tuple(np.around(box3D_point_in_image_6).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_3).astype(int)),
             tuple(np.around(box3D_point_in_image_7).astype(int)), (255, 0, 0), 1)
    cv2.line(image, tuple(np.around(box3D_point_in_image_4).astype(int)),
             tuple(np.around(box3D_point_in_image_8).astype(int)), (255, 0, 0), 1)

def main(client,
         image_path,
         label_path,
         lidar_path,
         depth_path,
         calib_path,
         valid_path,
         draw_point_cloud,
         draw_2D_box,
         draw_3D_box,
         show_label
         ):

    # set camera name and image type to request images and detections
    camera_name = "0"
    image_type = airsim.ImageType.Scene

    depth_camera_name = 'depth'
    depth_map_type = airsim.ImageType.DepthPerspective

    client.simSetCameraFov(camera_name=camera_name, fov_degrees=54.5, vehicle_name='PhysXCar')
    client.simSetCameraFov(camera_name=depth_camera_name, fov_degrees=54.5, vehicle_name='PhysXCar')

    # set detection radius in [cm]
    client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100, vehicle_name='PhysXCar')
    # add desired object name to detect in wild card/regex format
    client.simAddDetectionFilterMeshName(camera_name, image_type, "trafficone*", vehicle_name='PhysXCar')

    view_rect = ViewRect(0, 0, 1280, 720)
    scale = 1 / 100
    lidar_range = 100

    counter = 0

    print("Data recoding will begin after 5 seconds")
    time.sleep(5)

    while True:
        client.simPause(True)
        # get camera images and depth map from the car
        responses = client.simGetImages([
            airsim.ImageRequest(camera_name, image_type, False, False),  # scene vision image in png format
            airsim.ImageRequest(depth_camera_name, depth_map_type, True),  # depth in perspective projection
            ])

        # print('Retrieved images: %d' % len(responses))
        # make sure to get image and depth map at same time
        if len(responses) < 2:
            continue

        for response_idx, response in enumerate(responses):
            if response.image_type == depth_map_type:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                depth_in_meter = airsim.get_pfm_array(response)
                depth_in_millimeter = depth_in_meter * 1000

                # Convert depth_img to millimeters to fill out 16bit unsigned int space (0..65535).
                # Also clamp large values (e.g. SkyDome) to 65535
                depth_16bit = np.clip(depth_in_millimeter, 0, 65535).astype('uint16')
            elif response.image_type == image_type:
                # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                png = img1d.reshape(response.height, response.width, 3)
                png_visualization = np.copy(png)


        # Get Lidar data for one frame
        lidarData = client.getLidarData('Lidar1', 'PhysXCar')

        # Get detection info
        cones = client.simGetDetections(camera_name, image_type, vehicle_name='PhysXCar')

        # # Get lidar info
        # lidar_orientation = lidarData.pose.orientation
        # lidar_r = R.from_quat([lidar_orientation.x_val, lidar_orientation.y_val, lidar_orientation.z_val,
        #                        lidar_orientation.w_val])
        # lidar_rotation_mat = lidar_r.as_matrix()
        # lidar_position = np.array([lidarData.pose.position.x_val, lidarData.pose.position.y_val, lidarData.pose.position.z_val])

        if len(lidarData.point_cloud) > 3 and cones:
            transition_matrix = np.array(cones[0].transition_matrix.matrix)
            rotation_matrix = np.array(cones[0].rotation_matrix.matrix)
            projection_matrix = np.array(cones[0].projection_matrix.matrix)
            # NORTH - EAST - UP to EAST - UP - NORTH
            neu_to_eus = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            view_rotation_matrix = rotation_matrix.dot(neu_to_eus)
            view_projection_matrix = np.dot(transition_matrix, view_rotation_matrix).dot(projection_matrix)

            data_labels = []
            for cone in cones:
                label = KITTI_data_format()

                # Type
                cone_name = cone.name
                if 'yellow' in cone_name:
                    type = 'Yellow'
                elif 'blue' in cone_name:
                    type = 'Blue'
                else:
                    type = 'Orange'

                label.type = type

                # Get position of 2D bounding box
                left = cone.box2D.min.x_val
                top = cone.box2D.min.y_val
                right = cone.box2D.max.x_val
                bottom = cone.box2D.max.y_val

                label.left = round(left, 2)
                label.top = round(top, 2)
                label.right = round(right, 2)
                label.bottom = round(bottom, 2)

                # Get center point and its extend
                center_x, center_y, center_z = cone.center_point.x_val, cone.center_point.y_val, cone.center_point.z_val
                extend_x, extend_y, extend_z = cone.extend.x_val, cone.extend.y_val, cone.extend.z_val

                # Convert center point to camera space from world space
                center_point_world = np.array([center_x, center_y, center_z, 1])
                center_point_camera = center_point_world.dot(np.dot(transition_matrix, view_rotation_matrix))
                label.x = round(center_point_camera[0] * scale, 2)
                label.y = round(-center_point_camera[1] * scale, 2)
                label.z = round(center_point_camera[2] * scale, 2)

                scaled_extend_x, scaled_extend_y, scaled_extend_z = round(extend_x * scale, 2), round(extend_y * scale) \
                    , round(extend_z * scale, 2)

                label.height = scaled_extend_z * 2
                label.length = scaled_extend_x * 2
                label.width = scaled_extend_y * 2

                scaled_center_x, scaled_center_y, scaled_center_z = round(center_x * scale), round(center_y * scale)\
                    , round(center_z * scale, 2)
                distance = math.sqrt((scaled_center_x - lidarData.pose.position.x_val) ** 2 + (scaled_center_y - lidarData.pose.position.y_val) ** 2 + (scaled_center_z - (-lidarData.pose.position.z_val)) ** 2)
                if distance > lidar_range:
                    # set type
                    label.type = 'DontCare'

                    label.height = -1
                    label.width = -1
                    label.length = -1

                    label.x = -1000
                    label.y = -1000
                    label.z = -1000

                # Get rotation y
                r = R.from_quat([cone.relative_pose.orientation.x_val, cone.relative_pose.orientation.y_val,
                                 cone.relative_pose.orientation.z_val, cone.relative_pose.orientation.w_val])
                euler_xyz = r.as_euler('xyz', degrees=True)
                rotation_degree = round(euler_xyz[2], 2)
                rotation_mat = np.array([[math.cos(math.radians(rotation_degree)), -math.sin(math.radians(rotation_degree)), 0],
                                [math.sin(math.radians(rotation_degree)), math.cos(math.radians(rotation_degree)), 0],
                                [0, 0, 1]])

                euler_z = -round(euler_xyz[2], 2) - 90
                rotation_y = euler_z if np.abs(euler_z) <= 180 else euler_z + 360

                label.rotation_y = round(math.radians(rotation_y), 2)

                # Calculate alpha
                camera_vector = np.array([1, 0])
                cone_vector = np.array(
                    [round(cone.relative_pose.position.x_val, 2), round(cone.relative_pose.position.y_val, 2)])
                cos_angle = camera_vector.dot(cone_vector) / (
                            np.linalg.norm(camera_vector) * np.linalg.norm(cone_vector))
                angle = -np.arccos(cos_angle) if cone_vector[1] > 0 else np.arccos(cos_angle)
                alpha = rotation_y - angle

                label.alpha = round(math.radians(alpha), 2)

                data_labels.append(label)

                # draw the 2D bounding box
                if draw_2D_box:
                    draw2DBoundingBoxInImage(png_visualization, left, top, right, bottom)

                # show label in image
                if show_label:
                    showLabelInImage(png_visualization, left, top, cone.name)

                # draw the 3D bounding box
                if draw_3D_box:
                    draw3DBoundingBoxInImage(png_visualization, center_x, center_y, center_z, extend_x, extend_y, extend_z,
                                             rotation_mat, view_projection_matrix, view_rect)

            # handle point cloud
            point_cloud = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
            point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 3), 3))
            homogeneous_point_cloud = np.insert(point_cloud, 3, 1, 1)
            lidar_matrix = transition_matrix.dot(rotation_matrix)
            lidar_space_point_cloud = np.delete(homogeneous_point_cloud.dot(lidar_matrix), -1, 1) * scale

            if draw_point_cloud:
                projected_points = []
                for point in point_cloud:
                    projected_point = calculateProjectWorldToScreen(view_projection_matrix, point, view_rect)
                    if projected_point is not False:
                        projected_points.append(projected_point)

                # draw the point into image
                for projected_point in projected_points:
                    if projected_point[0] >= view_rect.min_x and projected_point[1] >= view_rect.min_y and projected_point[0] <= view_rect.max_x and projected_point[1] <= view_rect.max_y:
                        cv2.circle(png_visualization, tuple(np.around(projected_point).astype(int)), 1, (0, 0, 255), 0)

            # if cv2.waitKey(1) & 0xFF == ord('e'):
            # save data

            try:
                # save image data
                cv2.imwrite(image_path + "/%06d.png" % counter, png)

                # save depth map
                cv2.imwrite(depth_path + "/%06d.png" % counter, depth_16bit)

                # save label
                with open(label_path + "/%06d.txt" % counter, "w") as f:
                    for idx, label in enumerate(data_labels):
                        if idx == len(data_labels) - 1:
                            f.write(label.__str__())
                        else:
                            f.write((label.__str__() + "\n"))

                # save calib
                calib_str = 'R0_rect:'
                for item in np.nditer(projection_matrix):
                    calib_str += (' ' + str(item))
                calib_str += "\nTr_velo_to_cam:"
                for item in np.nditer(neu_to_eus):
                    calib_str += (' ' + str(item))
                with open(calib_path + "/%06d.txt" % counter, "w") as f:
                    f.write(calib_str)

                # save lidar data
                np.save(lidar_path + '/%06d.npy' % counter, lidar_space_point_cloud)

                # save valid image
                cv2.imwrite(valid_path + "/%06d.png" % counter, png_visualization)

                print("Data saving success, data index: %06d" % counter)
            except:
                print("An exception occurred when saving data")
            finally:
                counter += 1

        # Show image
        cv2.namedWindow("AirSim", 0)
        cv2.resizeWindow("AirSim", 1280, 720)
        cv2.imshow("AirSim", png_visualization)
        # cv2.resizeWindow("depth", 1280, 720)
        # cv2.imshow("depth", depth_16bit)

        client.simPause(False)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name: default})

    parser.add_argument('--data_path', type=str, default='../data', help='folder of data generation')
    parser.add_argument('--image_path_name', type=str, default='image', help='folder name of image data')
    parser.add_argument('--label_path_name', type=str, default='label', help='folder name of data label')
    parser.add_argument('--lidar_path_name', type=str, default='velodyne', help='folder name of lidar data')
    parser.add_argument('--depth_map_path_name', type=str, default='depth', help='folder name of depth map')
    parser.add_argument('--calib_path_name', type=str, default='calib', help='folder name of calibration matrix')
    parser.add_argument('--valid_path_name', type=str, default='valid', help='folder name of valid image')

    add_bool_arg('draw_point_cloud', default=True, help='whether to draw point cloud in image')
    add_bool_arg('draw_2D_box', default=False, help='whether to draw 2D bounding box in image')
    add_bool_arg('draw_3D_box', default=True, help='whether to draw 3D bounding box in image')
    add_bool_arg('show_label', default=True, help='whether to show object label in image')

    time_str = time.strftime("%Y%m%d%H%M", time.localtime())
    opt = parser.parse_args()
    save_data_path = os.path.join(opt.data_path, time_str)
    image_path = os.path.join(save_data_path, opt.image_path_name)
    label_path = os.path.join(save_data_path, opt.label_path_name)
    lidar_path = os.path.join(save_data_path, opt.lidar_path_name)
    depth_path = os.path.join(save_data_path, opt.depth_map_path_name)
    calib_path = os.path.join(save_data_path, opt.calib_path_name)
    valid_path = os.path.join(save_data_path, opt.valid_path_name)

    for path in [image_path, label_path, lidar_path, depth_path, calib_path, valid_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    np.set_printoptions(threshold=np.inf)

    main(client,
         image_path,
         label_path,
         lidar_path,
         depth_path,
         calib_path,
         valid_path,
         opt.draw_point_cloud,
         opt.draw_2D_box,
         opt.draw_3D_box,
         opt.show_label
    )