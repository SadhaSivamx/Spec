import cv2
import numpy as np
import math
from scipy.signal import find_peaks
Result=[]
def Aframe(img, val, x, y, w, h):
    return img[y-val:y+h+val, x-val:x+w+val, :]
def DrawBestCircle(original_img, cropped_image, x, y, thr,cond):
    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30,
                               minRadius=0, maxRadius=0, minDist=5)

    # Check if any circles were found
    if circles is not None:
        # Convert circles to integer values
        circles = np.uint16(np.around(circles))
        best_circle = circles[0][0]
        cx, cy, rad = best_circle
        adjusted_x = cx + x - thr
        adjusted_y = cy + y - thr
        if cond=="x":
            cv2.circle(original_img, (adjusted_x, adjusted_y), rad, (0, 255, 0), 2)
            cv2.line(original_img, (adjusted_x, adjusted_y), (adjusted_x+rad, adjusted_y),(255, 0, 0), 1)
            #cv2.putText(original_img, "Outer Diameter : {}mm".format(round(rad * 0.1, 2)),(20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            return ((adjusted_x, adjusted_y),cond,rad)
        else:
            cv2.circle(original_img, (adjusted_x, adjusted_y), rad, (0, 255, 0), 2)
            cv2.line(original_img, (adjusted_x, adjusted_y), (adjusted_x , adjusted_y+ rad), (255, 0, 0), 1)
            #cv2.putText(original_img, "Inner Diameter : {}mm".format(round(rad * 0.1, 2)), (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            return ((adjusted_x, adjusted_y),cond,rad)

def DrawBestCirclefrNut(original_img, cropped_image, x, y, thr):
    # Convert the cropped image to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, param1=100, param2=30,
                               minRadius=0, maxRadius=0, minDist=5)

    # Check if any circles were found
    if circles is not None:
        # Convert circles to integer values
        circles = np.uint16(np.around(circles))
        best_circle = circles[0][0]
        cx, cy, rad = best_circle
        adjusted_x = cx + x + thr
        adjusted_y = cy + y + thr
        cv2.circle(original_img, (adjusted_x, adjusted_y), rad, (0, 255, 0), 2)
        cv2.line(original_img, (adjusted_x, adjusted_y), (adjusted_x+rad, adjusted_y),(255, 0, 0), 1)
        return ((adjusted_x, adjusted_y),"x", rad)
def WasherParams(Fr,x,y,w,h):
    cropped_img = Aframe(Fr, 10, x, y, w, h)
    p,thr1,r1=DrawBestCircle(Fr, cropped_img, x, y, 10, "x")
    cropped_img = Aframe(Fr, -30, x, y, w, h)
    p,thr2,r2=DrawBestCircle(Fr, cropped_img, x, y, -30, "y")
    print(f"OD:{r1*0.1}mm ID:{r2*0.1}mm")
    cv2.putText(Fr,f"OD: {round(r1*0.1,2)} mm ID: {round(r2*0.1,2)} mm",(p[0]+10,p[1]-10), cv2.FONT_HERSHEY_SIMPLEX,0.3, (0, 0, 255), 1)
    Result.append(
        ["WASHER", f"OD : {round(r1*0.1,2)}mm", f"ID : {round(r2*0.1,2)} mm"])
# Load the image
def NutParams(Fr,x,y,w,h):
    cropped_img = Aframe(Fr, -10, x, y, w, h)
    p, thr1, rad = DrawBestCirclefrNut(Fr, cropped_img, x, y,10)
    return rad

def BoltParams(Fr,x,y,w,h):

    Frame = Aframe(Fr, 10, x, y, w, h)
    image = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, thresholded = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(thresholded, 50, 150)
    edges=cv2.dilate(edges,(3,3))
    # Apply Hough Transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)
    # Variables to store the longest line
    max_length = 0
    best_line = None

    # Find the longest line
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Calculate line length
            if length > max_length:
                max_length = length
                best_line = (x1+x, y1+y, x+x2, y2+y)
    return max_length,best_line
def PredProduct(Frx):
    def calculate_angle(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        delta_y = y2 - y1
        delta_x = x2 - x1
        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def angleGraph(ctr):
        tr,pre,co=30,0,0
        for i in range(len(ctr) - 1):
            angle = calculate_angle(ctr[i], ctr[i + 1])
            if abs(angle - pre) > tr:
                co += 1
            pre = angle
        print(f'Significant angle changes: {co}')
        return co

    gray = cv2.cvtColor(Frx, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area and area>500:
            max_area = area
            max_contour = contour

    if max_contour is not None:
        print(f'Maximum Area: {max_area}')
    a=angleGraph(max_contour[:, 0][::5])
    return a

cap = cv2.VideoCapture(1)  # Change the argument to the correct camera index if needed

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
color = (0, 255, 255)  # Black color in BGR
thickness = 1

def func(Frame):
    global Result
    image = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # The rest of your processing code...
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            # Store rectangle and contour
            a = ans = PredProduct(Frame[y - 10:y + h + 10, x - 10:x + w + 10, :])
            if 0 <= a <= 3:
                cv2.drawContours(Frame, contour, -1, (0, 255, 0), 2)
                Object = "Washer"
                WasherParams(Frame, x, y, w, h)
            elif a <= 10:
                cv2.drawContours(Frame, contour, -1, (0, 255, 0), 2)
                Object = "Nut"
                rad = NutParams(Frame, x, y, w, h)
                random_index = np.random.randint(len(contour))
                random_point = contour[random_index][0]  # contour points are in an array
                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                else:
                    center_x, center_y = 0, 0
                cv2.line(Frame, (center_x, center_y), random_point, (255, 0, 0), 1)
                distance = math.sqrt(((center_x - random_point[0]) ** 2) + ((center_y - random_point[1]) ** 2))
                cv2.circle(Frame, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.putText(Frame, f"Tc : {round((distance - rad) * 0.1, 2)}mm Rd : {round(rad * 0.1, 2)}mm",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                Result.append(
                    ["Nut", f"Thickness : {round((distance - rad) * 0.1, 2)}mm", f"Rd : {round(rad * 0.1, 2)}mm"])
            else:
                Object = "Bolt"
                max_length, ans = BoltParams(Frame, x, y, w, h)
                # Calculate peak-to-peak distances
                # Draw the longest line on the image and print coordinates
                if ans is not None:
                    x1, y1, x2, y2 = ans
                    print(f"Longest line coordinates: ({x1}, {y1}) to ({x2}, {y2})")

                    # Sample brightness values along the line
                    num_samples = max_length
                    brightness_values = []

                    for i in range(int(num_samples)):
                        alpha = i / num_samples  # Interpolate between the two points
                        xx = int(x1 * (1 - alpha) + x2 * alpha)
                        yy = int(y1 * (1 - alpha) + y2 * alpha)

                        # Ensure coordinates are within image bounds
                        if 0 <= xx < image.shape[1] and 0 <= yy < image.shape[0]:
                            brightness_values.append(image[yy, xx])  # Get pixel brightness

                    # Given brightness values

                    peaks, _ = find_peaks(brightness_values)
                    peak_distances = np.diff(peaks)
                cv2.line(Frame, (ans[0], ans[1]), (ans[2], ans[3]), (255, 0, 0), 2)
                cv2.putText(Frame,
                            f"Thread length : {round(max_length * 0.1, 2)}mm Pitch : {round(max(peak_distances) * 0.1, 2)}mm",
                            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                Result.append(["BOLT", f"Tlength : {round(max_length * 0.1, 2)}mm",
                                f"Pitch : {round(max(peak_distances) * 0.1, 2)}mm"])
            cv2.putText(Frame, f"B:orange N:Blue W:Yellow", (10, 460), font, font_scale, color, thickness)
            if Object == "Bolt":
                cv2.rectangle(Frame, (x, y), (x + w, y + h), (80, 127, 255), 2)
            elif Object == "Nut":
                cv2.rectangle(Frame, (x, y), (x + w, y + h), (237, 149, 100), 2)
            elif Object == "Washer":
                cv2.rectangle(Frame, (x, y), (x + w, y + h), (102, 255, 255), 2)
    # Display the result
    copy=Result.copy()
    Result=[]
    return Frame,copy
