import argparse
import logging
import time
import os
import cv2
import numpy as np

import face_recognition

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


logger = logging.getLogger('Pose-Live')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
path, dirs, files = next(os.walk("./output_imgs"))
files_no = len(files)
count = files_no


fps_time = 0


# Load a sample picture and learn how to recognize it.
nanni_image = face_recognition.load_image_file("/home/nd/Pictures/IMG_20181118_193724.jpg")
nanni_face_encoding = face_recognition.face_encodings(nanni_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

process_this_frame = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=str, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        ########################
        # Face recognition
        ########################

        # Resize image of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        # Only process every other frame of video to save time
        if process_this_frame:
            logger.debug('in frame+')
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match_level = face_recognition.face_distance([nanni_face_encoding], face_encoding)
                match = face_recognition.compare_faces([nanni_face_encoding], face_encoding)
                name = "Unknown"

                if match[0]:
                    name = "Nannigalaxy"
                    print("Matching percentage with Nanni is {0}%".format(round((1 - match_level[0]) * 100, 2)))

                    # print("Matching percentage with Nanni is {0}%".format(round((1-match_level[1])*100, 2)))
                else:
                    print("Matching percentage with Nanni (Unknown) is {0}%".format(round((1 - match_level[0]) * 100, 2)))

                logger.debug('face rec+')
                face_names.append(name)

        process_this_frame = not process_this_frame

        ########################
        # Posture
        ########################

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('human_behaviour_analysis result', image)
        # cv2.imwrite('./output_imgs/img' + str(count) + '.jpg',image)
        count += 1
        fps_time = time.time()
    # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        logger.debug('finished+')
    image.release()
    cv2.destroyAllWindows()
