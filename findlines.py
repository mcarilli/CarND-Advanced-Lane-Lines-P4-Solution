# For processing video files
from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import functions as fn

# Compute distortion coefficients from a known list of files
ret, mtx, dist, rvecs, tvecs  = fn.compute_distortion_coefs()

# Image processing pipeline function, to be passed to video-processing
# function later
def pipeline( img ):
    # for i in range(1,7):
    # filename = "test_images/test" + str(5) + ".jpg"

    # print( filename )

    # img = mpimg.imread(filename)

    # Undistort image using the distortion coefficients 
    # computed earlier
    undist = cv2.undistort( img, mtx, dist, None, mtx )

    sizex = undist.shape[1]
    sizey = undist.shape[0]
    img_size = [sizex, sizey]

    # Apply gradient magnitude, gradient direction, lightness,
    # and saturation thresholds to the undistorted input image.
    # Return "filtered," a binary image in which all pixels 
    # captured by the filter are marked as 1s
    S, filtered = fn.apply_filters( undist,
                                    thresh_lgt=40,
                                    thresh_ang=(0.8,1.2),
                                    thresh_mag=(40,255), 
                                    thresh_sat=(100,255)  )

    # Perspective transform:  source points
    srcbotright = [1121,sizey]
    srcbotleft = [209,sizey]
    srctopleft = [592,450]
    srctopright = [692,450]
    src = np.float32( [srcbotright, srcbotleft, srctopleft, srctopright] )

    # Perspective transform:  destination points 
    dstbotright = [srcbotright[0],sizey]
    dstbotleft = [srcbotleft[0],sizey]
    dsttopleft = [srcbotleft[0],0]
    dsttopright = [srcbotright[0],0]
    dst = np.float32( [dstbotright, dstbotleft, dsttopleft, dsttopright] )

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Compute the inverse transform for later use
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the filtered binary image, and multiply by 255 to use it as a grayscale image
    warped = 255*cv2.warpPerspective( filtered, M, (sizex,sizey), flags=cv2.INTER_LINEAR )

    # Compute left and right parabolas fitted to detected pixels for the
    # left and right lane lines, respectively
    left_fit, right_fit, output = fn.fit_lane_lines( warped )
  
    # Draw the lane line region between the two fitted parabolas back onto the undistorted frame.
    unwarped_lane_region = fn.draw_unwarped_lane_region( warped, undist, Minv, left_fit, right_fit )

    # Commented-out lines below were used to create figures for writeup.
    
    # fig = plt.figure(figsize=(12,4))

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.title('Original (distorted) image')

    # plt.subplot(122)
    # plt.imshow(undist)
    # plt.title('Undistorted image')

    # plt.tight_layout()
    # plt.show()

    # fig = plt.figure(figsize=(5,13))

    # plt.subplot(511)
    # plt.imshow(filtered, cmap = 'gray')
    # plt.title('Filtered image')

    # filtered_for_lines = 255*np.dstack((filtered,filtered,filtered))
    # pts = np.array([srcbotright,srcbotleft,srctopleft,srctopright], np.int32)
    # pts = pts.reshape((-1,1,2))
    # cv2.polylines( filtered_for_lines,[pts],True,(255,0,0),thickness=2)

    # plt.subplot(512)
    # plt.imshow(filtered_for_lines)
    # plt.title('Region to define perspective warp')

    # warped_with_lines = cv2.warpPerspective( filtered_for_lines, M, (sizex,sizey), flags=cv2.INTER_LINEAR )
    # plt.subplot(513)
    # plt.imshow(warped_with_lines)
    # plt.title('Filtered image post-warp')

    # For optional plotting use, create a set of points that trace
    # the best-fit parabolas in pixel space.   I ended up not 
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    # x-coords of lane lines at bottom of image
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # plt.subplot(514)
    # plt.imshow(output)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.title('Lane lines detected by sliding windows')

    # plt.subplot(515)
    # plt.imshow(unwarped_lane_region)
    # plt.title('Lane region filled + warped back onto image')

    # plt.tight_layout()
    # plt.show()
    
    # Get radii of curvature in meters for left and right lane lines
    rl, rr = fn.get_radii_of_curvature( unwarped_lane_region, left_fit, right_fit )

    # Print radii of curvature on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText( unwarped_lane_region, "Left radius of curvature = {:4.1f} m".format(rl),
                 (200,100), font, 1.5, (255,255,255), 2, cv2.LINE_AA )
    cv2.putText( unwarped_lane_region, "Right radius of curvature = {:4.1f} m".format(rr),
                 (200,160), font, 1.5, (255,255,255), 2, cv2.LINE_AA )

    # Compute pixel distance the car is off center by taking the average of the two 
    # lane line positions at the bottom of the image and subtracting that average
    # from the midpoint pixel x-value, given by shape( unwarped_lane_region/2
    # 
    # x-coords of lane lines at bottom of image
    leftx = left_fitx[-1]
    rightx = right_fitx[-1]
    distance_off_center_pixels = unwarped_lane_region.shape[1]/2. - (rightx+leftx)/2.
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    distance_off_center_meters = xm_per_pix*distance_off_center_pixels

    cv2.putText( unwarped_lane_region, "Distance off center = {:3.2f} m".format(distance_off_center_meters),
                 (200,220), font, 1.5, (255,255,255), 2, cv2.LINE_AA )

    # plt.imshow( unwarped_lane_region )
    # plt.show()

    return unwarped_lane_region

# Open the input video
clip = VideoFileClip('project_video.mp4')
# Process the input video to create the output clip
output_clip = clip.fl_image( pipeline )
# Write the output clip
output_clip.write_videofile( 'project_output.mp4', audio=False)
