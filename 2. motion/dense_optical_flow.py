import cv2
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), _ in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def motion_tracking():
    cap = cv2.VideoCapture("meme.mov")

    _, frame1 = cap.read()
    resize_dim = 600
    max_dim = max(frame1.shape)
    scale = resize_dim / max_dim
    frame1 = cv2.resize(frame1, None, fx=scale, fy=scale)
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while cap.isOpened():
        _, frame2 = cap.read()
        frame2 = cv2.resize(frame2, None, fx=scale, fy=scale)
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        rgb1 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        rgb = cv2.addWeighted(rgb1, 0.5, rgb2, 1, 0)

        #cv2.namedWindow('Optical flow', cv2.WINDOW_NORMAL)
        #cv2.imshow('Optical flow', draw_flow(next, flow))
        cv2.imshow('Optical flow', rgb)

        prev = next

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    motion_tracking()


if __name__ == '__main__':
    main()
