import os
import sys
import json
import cv2
import numpy as np
import pandas as pd

def clip_image_around_bbox_buffer(image, bbox, buffer=100):
    """
    Clips an image around a bounding box with a buffer.

    Args:
        image (ndarray): Input image.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        buffer (int): Buffer size.

    Returns:
        ndarray: Clipped image.
    """
    x1, y1, x2, y2 = bbox
    x1 -= buffer
    y1 -= buffer
    x2 += buffer
    y2 += buffer
    x1_ = max(0, int(x1))
    y1_ = max(0, int(y1))
    x2_ = min(int(x2), image.shape[1])
    y2_ = min(int(y2), image.shape[0])
    clipped_image = image[y1_:y2_, x1_:x2_]
    return clipped_image

def main(images_dir, annotation_json, side, out_dir):  

    with open(annotation_json) as f:
        annotation_json = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    
    cntr = 0

    cumulative_annos = []
    
    for t in annotation_json['images']:
        tf = t['file_name']
        ti = t['id']
        im = cv2.imread(f"{images_dir}/{tf}")
        box_num = 0
        
        for gt in annotation_json['annotations']:
            if gt['image_id'] == ti:
                bb = gt['bbox']
                bbxyxy = (bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3])
                bbox_data = list(bbxyxy)
                try:
                    clipped_img = clip_image_around_bbox_buffer(im, bbox_data)
                    if len(clipped_img) > 0:

                        #image_clipped_output_dirs = tf.split('/')
                        #os.makedirs(os.path.join(out_dir, image_clipped_output_dirs[0]), exist_ok=True)
                        #os.makedirs(os.path.join(out_dir, image_clipped_output_dirs[0], image_clipped_output_dirs[1]), exist_ok=True)
                        subdir = os.path.dirname(tf)
                        if subdir:
                            os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

                        if os.path.exists(os.path.join(out_dir, tf)):
                            # Multiple boxes present in image
                            box_num=box_num+1
                            image_boxes_categories = {"img_id": gt['image_id'], "file_name": tf, "box": bbox_data, "image_name_clip": f"{tf[:-4]}_{box_num}.jpg", 
                                                      "ann_id": gt['id'], 
                                                      "quality": gt['attributes']['building_quality'],
                                                      "type": gt['attributes']['building_type'],
                                                      "soft": gt['attributes']['has_soft_story'],
                                                      "story": gt['attributes']['num_of_stories'],
                                                      "overhang": gt['attributes']['overhang_type']}
                            #print(image_boxes_categories)
                            cumulative_annos.append(image_boxes_categories)
                            cv2.imwrite(os.path.join(out_dir, f"{tf[:-4]}_{box_num}.jpg"), clipped_img)
                        else:
                            image_boxes_categories = {"img_id": gt['image_id'], "file_name": tf, "box": bbox_data, "image_name_clip": f"{tf}", 
                                                      "ann_id": gt['id'], 
                                                      "quality": gt['attributes']['building_quality'],
                                                      "type": gt['attributes']['building_type'],
                                                      "soft": gt['attributes']['has_soft_story'],
                                                      "story": gt['attributes']['num_of_stories'],
                                                      "overhang": gt['attributes']['overhang_type']}
                            #print(image_boxes_categories)
                            cumulative_annos.append(image_boxes_categories)
                            cv2.imwrite(os.path.join(out_dir, f"{tf}"), clipped_img)
                        cntr = cntr+1
                        print("Annotation count: ", cntr)
                except Exception as e:
                    # Print the error message if an exception occurs
                    print("An error occurred:", e, im, bbox_data)
    df_out = pd.DataFrame(cumulative_annos)
    df_out["weights"] = 1
    df_out.to_csv(f"{out_dir}/cumulative_annos_{side}.csv")

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc == 5:
        IMG_DIR, ANN_JSON, SIDE, OUT_DIR = sys.argv[1:5]
    elif argc == 4:
        IMG_DIR, ANN_JSON, OUT_DIR = sys.argv[1:4]
        SIDE = "default"
        print("[INFO] SIDE not provided — using SIDE='default'")
    else:
        print("Usage: python prep_classifier_training_data.py <IMG_DIR> <ANN_JSON> [<SIDE>] <OUT_DIR>")
        sys.exit(1)

    main(IMG_DIR, ANN_JSON, SIDE, OUT_DIR)
