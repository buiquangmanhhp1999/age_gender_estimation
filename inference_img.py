import numpy as np
import cv2
import time
import data_utils as utils
from model import Efficient_Net
import config as cf
from mtcnn import MTCNN

if __name__ == '__main__':
    # Set up arguments
    arguments = utils.get_arguments()
    margin = arguments.margin
    img_path = arguments.image_path
    weight_path = arguments.weight_file
    boxes = []

    # Detector use by MTCNN
    detector = MTCNN()

    # Load image
    image = cv2.imread(img_path)
    img_h, img_w = image.shape[:2]

    # Load weights
    model = Efficient_Net(trainable=False).model
    model.load_weights(weight_path)
    
    # Run detector through a image(frame)
    start_time = time.time()
    result = detector.detect_faces(image)
    end_time = time.time()

    num_detected_face = len(result)
    if num_detected_face == 0:
        print('No detected face')
        exit()

    faces = np.empty((num_detected_face, cf.IMAGE_SIZE, cf.IMAGE_SIZE, 3))
    print("{}: detected {} faces on {}s".format(img_path, num_detected_face, end_time - start_time))

    # crop faces
    for i in range(len(result)):
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']

        # coordinates of boxes
        left, top = bounding_box[0], bounding_box[1]
        right, bottom = bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]

        # coordinates of cropped image
        x1_crop = max(int(left), 0)
        y1_crop = max(int(top), 0)
        x2_crop = int(right)
        y2_crop = int(bottom)

        cropped_face = image[y1_crop:y2_crop, x1_crop:x2_crop, :]
        face = cv2.resize(cropped_face, (cf.IMAGE_SIZE, cf.IMAGE_SIZE))
        faces[i, :, :, :] = face
        box = (x1_crop, y1_crop, x2_crop, y2_crop)
        boxes.append(box)

    # predict
    result = model.predict(faces / 255.0)

    # Draw bounding boxes and labels on image
    image = utils.draw_labels_and_boxes(image, boxes, result, margin)

    if image is None:
        exit()

    image = cv2.resize(image, (img_w, img_h), cv2.INTER_AREA)
    cv2.imwrite('selenagomez.jpg', image)
    cv2.imshow('img', image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()