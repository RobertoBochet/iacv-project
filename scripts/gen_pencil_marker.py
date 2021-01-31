import cv2

dictionary = cv2.aruco.custom_dictionary(5, 3)
img = cv2.aruco.drawMarker(dictionary, 0, 400)
cv2.imwrite('./markers/pencil_marker.jpg', img)
