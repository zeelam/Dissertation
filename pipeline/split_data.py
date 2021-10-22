import os

def write_txt(txt_name, annotated_path, data_path):
    i = 0
    with open(os.path.join(root_path, txt_name), "w") as f:
        for dir in os.listdir(annotated_path):
            # get left camera depth
            left_depth_path = os.path.join(annotated_path, dir, "proj_depth", "groundtruth", "image_02")

            for annotated_image in os.listdir(left_depth_path):
                # get raw image
                date = dir[:10]
                data_stored_path = os.path.join(data_path, date, dir, "image_02", "data")
                if os.path.exists(os.path.join(data_stored_path, annotated_image)):
                    if i != 0:
                        train_txt = "\n"
                    else:
                        train_txt = ""
                    train_txt += os.path.join(os.path.join(date, dir, "image_02", "data"), annotated_image)
                    train_txt += " "
                    train_txt += os.path.join(os.path.join(dir, "proj_depth", "groundtruth", "image_02"),
                                              annotated_image)
                    f.writelines(train_txt)
                    i += 1

if __name__ == '__main__':
    current_path = os.getcwd()
    root_path = os.path.join(current_path, "../data/kitti")
    annotation_path = os.path.join(root_path, "depth_annotation")
    annotated_train_path = os.path.join(annotation_path, "train")
    annotated_val_path = os.path.join(annotation_path, "val")
    data_path = os.path.join(root_path, "data")

    write_txt("eigen_train_files_with_gt.txt", annotated_train_path, data_path)
    write_txt("eigen_val_files_with_gt.txt", annotated_val_path, data_path)