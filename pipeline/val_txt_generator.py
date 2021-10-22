import os

if __name__ == '__main__':
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "../data/kitti")
    image_path = os.path.join(data_path, "data")
    test_file = os.path.join(data_path, "eigen_test_files_with_gt.txt")

    f = open(test_file, 'r')
    test_list = f.readlines()
    f.close()
    n_exist_data = 0
    with open(os.path.join(data_path, "eigen_val_files_with_gt.txt"), "w") as f:
        val_txt_str = ""
        for line in test_list:
            rgb_image_path = line.split(" ")[0]
            depth_map_path = line.split(" ")[1]
            if os.path.exists(os.path.join(image_path, rgb_image_path)):
                if n_exist_data == 0:
                    val_txt_str += (rgb_image_path + " " + depth_map_path)
                else:
                    val_txt_str += ("\n" + rgb_image_path + " " + depth_map_path)
                n_exist_data += 1
        f.writelines(val_txt_str)

