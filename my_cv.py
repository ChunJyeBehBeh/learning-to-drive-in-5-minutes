import cv2
import numpy as np

def remove_noise(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def discard_colors(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(image, low_threshold, high_threshold):
    return cv2.Canny(image, low_threshold, high_threshold)


def draw_lines(image, lines, color=[255, 0, 0], thickness=2):
    # print("Draw Lines {}".format(len(lines)))
    for line in lines:
        for x1,y1,x2,y2,slope in line:
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def slope(x1, y1, x2, y2):
    try:        
        return (y1 - y2) / (x1 - x2)
    except:
        return 0
        

def separate_lines(lines):
    right = []
    left = []

    if lines is not None:
        for x1,y1,x2,y2 in lines[:, 0]:
            m = slope(x1,y1,x2,y2)
            if m >= 0:
                right.append([x1,y1,x2,y2,m])
            else:
                left.append([x1,y1,x2,y2,m])
    return left, right


def reject_outliers(data, cutoff, threshold=0.08, lane='left'):
    data = np.array(data)
    # print("{}: {}".format(lane,data[:, 4]))
    # print(data)
    # print("-----")
    data = data[(data[:, 4] >= cutoff[0]) & (data[:, 4] <= cutoff[1])]
    # print(len(data))
    try:
        if lane == 'left':
            # return(data[find_outlier(data)])
            return data[np.argmin(data,axis=0)[-1]]
        elif lane == 'right':
            return data[np.argmax(data,axis=0)[-1]]
    except:
        print("Error")
        return []

def find_outlier(data,lane='left'):
    # Use area to determine the outer most line
    threshold_y = 5

    area_result = []
    
    if lane == 'left':
        for x1,y1,x2,y2,_ in data:
            area = (y2-y1)*(x1+x2)*0.5
            if abs(y2-y1)<threshold_y:
                area = 99999
            area_result.append(abs(area))
        return(np.argmin(area_result,axis=0))





