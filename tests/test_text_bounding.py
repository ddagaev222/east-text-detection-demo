import site

import cv2
import imutils
import numpy as np
import pytest

from east_demo import decode_predictions, non_max_suppression, rotated_Rectangle

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

site_path = [path for path in site.getsitepackages() if "site-packages" in path][0]

REGRESSION_TEST_CASES = [
    (
        site_path + "/data/image1.jpg",
        (
            np.array([[142, 130, 191, 143], [57, 108, 99, 121]]),
            np.array([0.24483512, 0.23821796]),
        ),
    ),
    (
        site_path + "/data/image2.png",
        (
            np.array([[33, 14, 233, 126], [17, 122, 235, 220]]),
            np.array([0.04301158, 0.02977725]),
        ),
    ),
]


def detect_text_bounding_box(frame):
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
    interpreter = Interpreter(
        model_path=site_path
        + "/model/east_text_detection_256x256_integer_quant.tflite",
        num_threads=4,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check to see if we have reached the end of the stream
    if frame is None:
        return None

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
        points = rotated_Rectangle(orig, rotatedRect, color=(0, 255, 0), thickness=2)
        cv2.polylines(
            orig,
            [points],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
            lineType=cv2.LINE_8,
            shift=0,
        )

    # show the output frame
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)

    return boxes, angles


@pytest.mark.parametrize("image, expected_result", REGRESSION_TEST_CASES)
def test_regression_text_detection(image, expected_result):
    frame = cv2.imread(image)

    if frame is None:
        pytest.fail(f"Test failed. Could not open file {image}.")

    result = detect_text_bounding_box(frame)
    print(result)
    assert len(result) == len(expected_result)
    for elm1, elm2 in zip(result, expected_result):
        assert np.all(elm1) == np.all(elm2)
