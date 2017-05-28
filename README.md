## The output video (road with lane line area marked) is project_output.mp4.

## findlines.py contains the video processing pipeline.

## functions.py contains utility functions.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[dist_and_undist]: ./output_images/dist_and_undist.png "Distorted and undistorted images"
[filters]: ./output_images/filters.png "Filters"
[pipeline]: ./output_images/pipeline.png "Pipeline"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is found in compute_distortion_coefs() at line 11 of functions.py, and called at line 12 of findlines.py.

I calibrated the camera by using a set of provided images taken with the the camera.  The images are pictures of chessboards.
For each image, I used cv2.findChessboardCorners() to locate the corners of squares (intersections between white and black squares),
 and appended the locations of these corners to an array "imgpoints."  For each image where corners were successfully found,
I also appended "objpoints" to another array.  Objpoints represents a set of regularly spaced locations where the corners should lie if
the image is dead center with a certain scaling.  By comparing the "objpoints" and "imgpoints" arrays, cv2.drawChessboardCorners()
is able to extract distortion coefficients for the camera. 

Extracting the distortion coefficients only needed to be done once, at the beginning of the program, prior to the processing pipeline.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

My pipeline first undistorted an input image (or video frame) by applying cv2.undistort() to the image, using the distortion coefficients
extracted above.

![Distorted and undistorted images][dist_and_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My function to apply filters and thresholds to create a binary image with suspected lane line pixels tagged as 1s is apply_filters(),
at line 49 of functions.py. Within the processing pipeline, apply_filters is invoked at line 36 of findlines.py.  
Please see functions.py for more details.

![Filters and resulting binary output][filters]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used cv2.getPerspectiveTransform() to find the transformation matrix at line 57 of 
findlines.py, then applied the transformation using cv2.warpPerspective() at line 61.

I used the following hardcoded source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| [1121, 720]   | [1121, 720] 
| [ 209, 720]   | [ 209, 720]
| [ 592, 450]   | [ 209,   0]
| [ 692, 450]   | [1121,   0]

This resulted in straight lane lines becoming approximately vertical in the transformed image.

An example of pre- and post-warp frames can be found in the image sequence after point 6. below.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels were identified in fit_lane_lines(), at line 151 of functions.py.  fit_lane_lines() calls find_window_centroids()
at line 110 of functions.py to perform the convolution search described below.  Within the processing pipeline, fit_lane_lines()
is called at line 65 of findlines.py.

I identified left and right lane line pixels in the warped, filtered binary image using a convolution technique.
First, a horizontal slab (the bottom quarter of the warped image) was summed vertically, and the resulting 1D array convolved with an array 
of 1s of tunable width (I used 150).  The convolution (also a 1D array) typically showed peaks at the locations of the left and 
right lane lines.  This procedure was repeated for the remaining three horizontal slabs.

I used a convolution width of 150 because near the top of the image, the perspective transform tended to
warp regions of suspected lane line pixels to 100+ pixels across.  Using this (generous) width did not incur significant risk 
of false detections, because my gradient and color filters applied earlier did a good job of filtering for lane line pixels alone.

I also implemented code to optionally skip the convolution step if a set of best-fit polynomials for each lane line already existed,
and simply tag lane line pixels within a certain width around each existing polynomial (then fit new polynomials to the new pixels).
However, I ended up not using this functionality in my final pipeline, since the pipeline worked on the first try 
using convolutions alone.

Once the left and right lane line pixels were identified using convolutions, the locations of the pixels found within those windows
were used to fit two quadratic polynomials, one each for the left and right lanes.

An example of the lane line regions identified by convolutions for a warped+filtered image, along with the corresponding
quadratic fits, can be seen in the image sequence below.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated by first generating a set of x-y locations along each lane line polynomial in pixel space,
converting these x-y coordinates to world space (by converting from pixel coordinates to meters), fitting a new polynomial to the
pixels in world space, and applying a formula from the lessons to the world-space polynomial coefficients at the car's location 
(bottom of of the image).  This is implemented in get_radii_of_curvature() at line 280 of functions.py.  Within the pipeline, 
get_radii_of_curvature() is invoked at line 128 of findlines.py. 

The position of the vehicle with respect to center is computed by calculating the center of the detected lane region at the bottom of
the image, as the average of the left and right polynomials evaluated at the bottom of the image.  This value is subtracted from
the width of the image/2 to obtain an off-center distance in pixels, which is then multiplied by the x-direction pixel-to-meters 
conversion factor of 3.7/700.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane line region is drawn, and warped back onto the undistorted image/video frame, in draw_unwarped_lane_region(), at line 248
of functions.py.  Within the pipeline, draw_unwarped_lane_region() is invoked at line 68 of findlines.py.

The following sequence of images shows the operation of my pipeline from the filtered image to drawing the final frame.

![Pipeline][pipeline]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](./project_output.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging part of this project was implementing a filter that reliably detected lane line pixels in varying lighting/background
conditions for both yellow and white lane lines.  I had to play with a number of different filters before I got something satisfactory.
In the end, a combination of saturation, lightness, gradient magnitude, and gradient direction gave reliable detection of lane lines 
on the road regions of the image.  See apply_filters() at line 49 of functions.py for details of my final filter choices.

My pipeline tends to fail at longer distances 
when the curvature of the road is too extreme, as can be seen in my video.  At times when the road curves
most sharply, it can be seen that near the faraway end of the highlighted region, the highlighted region loses track of one of the lanes.
This is because when the road curves too far outside the trapezoidal source region used for the perspective transform, the pixels that
veer too far from the trapezoidal region are warped out of the transformed image, and end up not being used in the convolution
search or polynomial fit.  This could be alleviated by using a less aggressive perspective transform (a wider trapezoidal
source region that gives the road more room to curve).

For the project video this loss of faraway lane line pixels due to warping is not a major issue, but it would be exacerbated 
for more sharply curving roads.
