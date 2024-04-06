import site
import tempfile
import time

import cv2
import imutils.video
import numpy as np
import streamlit as st
from imutils.video import FPS

from east_demo import decode_predictions, non_max_suppression, rotated_Rectangle

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def process_video(vs: cv2.VideoCapture):
    fpsstr = ""
    framecount = 0
    time1 = 0.0
    site_path = [path for path in site.getsitepackages() if "site-packages" in path][0]

    st.write("[INFO] Starting...")

    # initialize the original frame dimensions, new frame dimensions,
    # and ratio between the dimensions
    (W, H) = (None, None)
    (newW, newH) = (256, 256)
    (rW, rH) = (None, None)

    mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text

    # load the pre-trained EAST text detector
    st.write("[INFO] loading EAST text detector...")
    interpreter = Interpreter(
        model_path=site_path
        + "/model/east_text_detection_256x256_integer_quant.tflite",
        num_threads=4,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # start the FPS throughput estimator
    fps = FPS().start()

    frame_placeholder = st.empty()
    # loop over frames from the video stream
    while True:
        t1 = time.perf_counter()

        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        frame = vs.read()[1]

        # check to see if we have reached the end of the stream
        if frame is None:
            break

        # resize the frame, maintaining the aspect ratio
        frame = imutils.resize(frame, width=640)
        orig = frame.copy()

        # if our frame dimensions are None, we still need to compute the
        # ratio of old frame dimensions to new frame dimensions
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            rW = W / float(newW)
            rH = H / float(newH)

        # resize the frame, this time ignoring aspect ratio
        frame = cv2.resize(frame, (newW, newH))

        # construct a blob from the frame and then perform a forward pass
        # of the model to obtain the two output layer sets
        frame = frame.astype(np.float32)
        frame -= mean
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        interpreter.set_tensor(input_details[0]["index"], frame)
        interpreter.invoke()

        scores = interpreter.get_tensor(output_details[0]["index"])
        geometry1 = interpreter.get_tensor(output_details[1]["index"])
        geometry2 = interpreter.get_tensor(output_details[2]["index"])
        scores = np.transpose(scores, [0, 3, 1, 2])
        geometry1 = np.transpose(geometry1, [0, 3, 1, 2])
        geometry2 = np.transpose(geometry2, [0, 3, 1, 2])

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences, angles) = decode_predictions(scores, geometry1, geometry2)
        boxes, angles = non_max_suppression(
            np.array(rects), probs=confidences, angles=np.array(angles)
        )

        # loop over the bounding boxes
        for (startX, startY, endX, endY), angle in zip(boxes, angles):
            # scale the bounding box coordinates based on the respective ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # draw the bounding box on the frame
            width = abs(endX - startX)
            height = abs(endY - startY)
            centerX = int(startX + width / 2)
            centerY = int(startY + height / 2)

            rotatedRect = (
                (centerX, centerY),
                ((endX - startX), (endY - startY)),
                -angle,
            )
            points = rotated_Rectangle(
                orig, rotatedRect, color=(0, 255, 0), thickness=2
            )
            cv2.polylines(
                orig,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_8,
                shift=0,
            )
            cv2.putText(
                orig,
                fpsstr,
                (640 - 170, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (38, 0, 255),
                1,
                cv2.LINE_AA,
            )

        # update the FPS counter
        fps.update()

        # show the output frame
        frame_placeholder.image(orig, channels="BGR")

        # FPS calculation
        framecount += 1
        if framecount >= 10:
            fpsstr = "(Playback) {:.1f} FPS".format(time1 / 10)
            framecount = 0
            time1 = 0.0
        t2 = time.perf_counter()
        elapsedTime = t2 - t1
        time1 += 1 / elapsedTime

    # stop the timer and display FPS information
    fps.stop()
    st.write("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    st.write("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    st.title("EAST Text Detection Demo App")
    videofile = st.file_uploader("Upload a video file", type=["mov"])

    if videofile is not None:
        st.title(videofile.name)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(videofile.read())

        # Create a VideoCapture object from the byte buffer
        vs = cv2.VideoCapture(tfile.name)

        process_video(vs)
        st.success("Text Detection completed!")


if __name__ == "__main__":
    main()
