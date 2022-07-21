import cv2 as cv
import numpy as np
from math import atan2, cos, sin, sqrt, pi
from matplotlib import pyplot as plt
from scipy import ndimage

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    ## [visualization1]
    angle = atan2(q[0] - p[0], q[1] - p[1]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    ## [visualization1]

def maxContourIndex(allContours):
    allContours = np.array([np.array([i[0] for i in j]) for j in allContours]) 
    center_of_massAll = [list(ndimage.measurements.center_of_mass(i)) for i in allContours]
    print(center_of_massAll.index(max(center_of_massAll)))
    return center_of_massAll.index(max(center_of_massAll))

image = cv.imread("img1.bmp")
image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
g = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
edge = cv.Canny(g, 100, 255)
contours, hierarchy = cv.findContours(edge, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv.contourArea, reverse=True)
mci = maxContourIndex(contours)

# for simple shape
cnt = cv.convexHull(contours[mci])

cv.drawContours (image, [cnt], 0, (0, 255, 0), 2)

moments = cv.moments(cnt)
M = np.array([[moments['mu20'], moments['mu11']],[moments['mu11'],moments['mu02']]])
eigenvalues, eigenvectors = np.linalg.eig(M)
area = moments['m00']

mean_x = moments['m10'] / area

mean_y = moments['m01'] / area

p1 = (mean_x + 0.02 * eigenvectors[0,0] * eigenvalues[0], mean_y + 0.02 * eigenvectors[0,1] * eigenvalues[0])
p2 = (mean_x - 0.02 * eigenvectors[1,0] * eigenvalues[1], mean_y - 0.02 * eigenvectors[1,1] * eigenvalues[1])

cntr = (int(mean_x), int(mean_y))
cv.circle(image, cntr, 3, (255, 0, 255), 2)

drawAxis(image, cntr, p1, (0, 255, 0), 1)
drawAxis(image, cntr, p2, (255, 255, 0), 5)

angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
print("Угол поворота: " + str(int(np.rad2deg(angle))) + " градусов")

print("Собственные значения: " + str(eigenvalues) + ", собственные векторы: " + str(eigenvectors))

cv.imwrite('result.jpg', image)
 