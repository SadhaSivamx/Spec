### Spec

Spec is a server-based application used for the remote inspection of various mechanical parts, allowing for dimensional and geometrical analysis without the use of AI or ML.

###### Output 

![Project-Template](https://github.com/user-attachments/assets/bdbcb3fe-c95d-483f-b435-e8d830229d61)

### Working

|  Specimen | Operation  |
| ------------ | ------------ |
|  Bolt |   Canny Edge -> **Finding Best Line** -> **Finding Pitch** |
|   Washer |   **Finding Best Circle** -> Zoom & Inv -> **Finding Best Circle** | 
|   Nut |  **Finding Best Circle** -> Moment Centre -> Difference of Radius and Contour Point |

**Contour Shape Analysis**

Thresholding is done on the image and the contours are retrieved and further sent to a function which calculates the Angle between two consecutive points and if the angle between increases more than a certain threshold the  count is incremented . And the Final Value Helps us Predict the shape of the Object


**Finding Best Circle**

Its a very popular algorithm used for finding circles in an image , for more info you can refer to one of my [post](https://www.linkedin.com/posts/sadx2k5_houghs-circle-transform-activity-7230532969671577600-5Jy0) on this topic

**Finding Best Line**

Using Hough's Line Transform we find all the possible lines and keep track of the max distance Line by changing the value if the value is more than the preset value and for finding the brightness of pixels under the line we use interpolation. 

**Finding Pitch**

![Screenshot 2024-11-01 195715](https://github.com/user-attachments/assets/0ac968d8-a8d0-4f19-b34f-84d68fcdec5d)

From the brightness values retrieved we calculate the peaks and find the distance between them , we get the pitch value as pitch refers to the distance between peaks ( higher Brightness Value )

**Pixels to mm**

This can be done by calculating the ratio required to convert px to mm , calculate dimention for known  length then approximate the ratio of pixel to mm
