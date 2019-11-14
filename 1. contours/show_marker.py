import cv2
import numpy as np

BLUE_MIN = 100
BLUE_MAX = 130


def view_image(image, title="Image"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)


def show_marker(img, hue_min, hue_max):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([hue_min, 0, 0], np.uint8)
    upper = np.array([hue_max, 255, 200], np.uint8)
    binary_img = cv2.inRange(hsv_img, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    opened_img = cv2.cvtColor(opened_img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.rectangle(opened_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    concatenated = cv2.hconcat([img, opened_img])
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", concatenated)


def main():
    cap = cv2.VideoCapture('blue.mov')
    if not cap.isOpened():
        print('Error')

    while cap.isOpened():
        grabbed, frame = cap.read()
        if grabbed:
            show_marker(frame, BLUE_MIN, BLUE_MAX)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
