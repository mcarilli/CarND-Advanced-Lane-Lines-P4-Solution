import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)


# Compute distortion coefficients for this camera from a known list of 
# chessboard images.
def compute_distortion_coefs( nx=9, ny=6 ):
    objpoints = []
    imgpoints = []

    # Create array of destination points for undistorting
    objp = np.zeros( (nx*ny,3), np.float32 )
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # print( objp )

    for i in range(1,21):
        filename = "camera_cal/calibration" + str(i) + ".jpg"
        # Read chessboard image
        img = mpimg.imread(filename)

        # Convert image to grayscale
        gray = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )

        # Find the chessboard corners within the images
        ret, corners = cv2.findChessboardCorners( gray, (nx,ny), None )

        print(ret)

        if ret == True:
            imgpoints.append( corners )
            objpoints.append( objp )
            
            # Optional, for plotting/sanity check purposes:  draw the discovered 
            # chessboard corners on the input image
            img = cv2.drawChessboardCorners( img, (nx,ny), corners, ret )

    # iterate through gray.shape in reverse order
    # to obtain pixel dimensions expected by findChessboardCorners
    # Compute distortion coefficients using the located chessboard corners and the 
    # target locations ("objpoints") for those corners.
    return cv2.calibrateCamera( objpoints, imgpoints, gray.shape[::-1], None, None )

# Apply filters to locate suspected lane line pixels.
# I use a lightness filter, a saturation filter, a gradient magnitude filter, and a gradient angle filter.
# The final binary output is produced by combining them as follows:
# lightness filter & ( ( gradient mag filter & gradient angle filter ) | saturation filter ).
def apply_filters( img, sobel_kernel=3, 
                   thresh_lgt=40, # Lightness threshold (filter out dark pixels in HLS color space)
                   thresh_ang=(0,np.pi/2), # Angle threshold for gradient (only accept gradients within the right angle range)
                   thresh_mag=(0,255), # Magnitude threshold for gradients (only accept sufficiently strong gradients)
                   thresh_sat=(0,255) ): # Saturation threshold:  only accept pixels with a certain saturation in HLS color space
    # print( thresh_ang )
    # print( thresh_mag ) 
    # print( thresh_sat ) 

    # Create array of magnitude-of-gradient values, scaled between 0 and 255
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Compute x and y gradients using Sobel stencils
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absx = np.abs( sobelx )
    absy = np.abs( sobely )
    # Compute angle formed by absolute value of x and y gradients 
    # (will fall between 0 and pi/2)
    angles = np.arctan2( absy, absx )
    # Compute magnitude of gradient
    mag = np.sqrt(sobelx**2+sobely**2)
    scaled_mag = np.uint8( 255*mag/np.max( mag ) )
    # print(np.max(mag))
    # print(np.max(scaled_mag))

    # Create arrays of lightness and saturation values
    hls = cv2.cvtColor( img, cv2.COLOR_RGB2HLS )
    L = hls[:,:,1]
    S = hls[:,:,2]

    # fig = plt.figure(figsize=(10,12))

    # plt.subplot(321)
    # plt.imshow(img)
    # plt.title('Undistorted image')

    # plt.subplot(322)
    # plt.imshow( ( L > thresh_lgt ), cmap='gray' )
    # plt.title('Lightness filter')

    # plt.subplot(323)
    # plt.imshow( ( scaled_mag >= thresh_mag[0] ) & 
    #                  ( scaled_mag <= thresh_mag[1] ), cmap='gray' )
    # plt.title('Gradient magnitude filter')

    # plt.subplot(324)
    # plt.imshow( ( angles >= thresh_ang[0] ) &
    #             ( angles <= thresh_ang[1] ), cmap='gray' )
    # plt.title('Gradient angle filter')

    # plt.subplot(325)
    # plt.imshow( ( S >  thresh_sat[0] ) &
    #             ( S <= thresh_sat[1] ), cmap='gray')
    # plt.title('Saturation filter')

    # Create output image that will contain 1s for suspected lane line pixels and 0s elsewhere
    binary_output = np.zeros_like( scaled_mag )
    binary_output[ ( L > thresh_lgt ) &  # Apply lightness threshold filter
                   ( ( ( scaled_mag >= thresh_mag[0] ) & # Apply gradient magnitude filter
                     ( scaled_mag <= thresh_mag[1] ) &   
                     ( angles >= thresh_ang[0] ) &   # Apply gradient direction filter
                     ( angles <= thresh_ang[1] ) ) |
                     ( ( S >  thresh_sat[0] ) & # Apply saturation filter
                     ( S <= thresh_sat[1] ) ) ) ] = 1

    # plt.subplot(326)
    # plt.imshow(binary_output,cmap='gray')
    # plt.title('Binary output (combined filters)')

    # plt.tight_layout()
    # plt.show()

    return scaled_mag, binary_output

# Tunable parameters for window search
window_width = 150 
window_height = 180 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

# Return an array of ones where a rectangular mask is present, and zeros elsewhere. 
# Based on code from the quizzes.
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Find centroids of detected lane line pixel regions for left and right lane lines 
# by taking considering several horizontal slices of the array.  For each slice, 
# the pixels are summed vertically, and the resulting 1D array is convolved with a mask of tunable width.
# In principle, this convolution should have 2 peaks, one on the the left and one on the right, 
# corresponding the approximate centers of the left and right lane lines within that 
# horizontal slice.
# Based on code from the quizzes.
def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # Vertically sum this horizontal slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
            # Convolve the resulting 1D array with our window width
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids

# Utility function that returns coefficients of a quadratic polynomial fitted to the left and right 
# lane lines.
# Based on code from the quizzes.
def fit_lane_lines( warped, preexistingFit=False, left_fit=None, right_fit=None ):
    
    # Create RGB version of grayscale "warped" image
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8)
    # print( np.max( warpage ) )
   
    # If no polynomial fit from the last frame is given, do a sliding window search first to find 
    # lane line pixels, then fit a polynomial to those pixels.
    if( preexistingFit == False ): 
        window_centroids = find_window_centroids(warped, window_width, window_height, margin)
        
        # If we found any window centers
        if len(window_centroids) > 0:
        
            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)
        
            # Go through each level and draw the windows 	
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
        	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        	    # Add graphic points from window mask here to total pixels found 
        	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
        
            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channel
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            output = cv2.addWeighted(warpage, 0.5, template, 0.5, 0.0) # overlay the original road image with window results
         
        # If no window centers found, just display orginal road image
        else:
            output = warpage

        l_pixels_to_fit = np.zeros_like(warped)
        r_pixels_to_fit = np.zeros_like(warped)

        # print( warped.shape )
        # print( r_points.shape )

        l_pixels_to_fit[( warped > 0 ) & ( l_points > 0 )] = 1
        r_pixels_to_fit[( warped > 0 ) & ( r_points > 0 )] = 1

        # Extract x and y values of pixels associated with left and right lane lines.
        l_xvals_to_fit = l_pixels_to_fit.nonzero()[1] 
        l_yvals_to_fit = l_pixels_to_fit.nonzero()[0]
        r_xvals_to_fit = r_pixels_to_fit.nonzero()[1] 
        r_yvals_to_fit = r_pixels_to_fit.nonzero()[0]

        # Fit a quadratic to the left and right pixel locations, with y as the "x-coordinate" of the polynomial.
        left_fit = np.polyfit( l_yvals_to_fit, l_xvals_to_fit, 2)
        right_fit = np.polyfit( r_yvals_to_fit, r_xvals_to_fit, 2)

    # If a polynomial fit lane lines were found in the last frame, search near them for lane lines
    # instead of redoing the sliding window search.
    # 
    # This is optional, and actually I did not end up using it in my pipeline, because it worked on the first try
    # using a pure sliding-windows approach.
    else:
        nonzero = warped.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & 
                          (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & 
                           (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

    # ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # # Display the final results
    # plt.imshow(output)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.show()

    return left_fit, right_fit, output

# Draw the lane region (the region between the polynomials fitted to the 
# left and right lane lines) back onto the undistorted image or video frame.
def draw_unwarped_lane_region( warped, undist, Minv, left_fit, right_fit ):
    # Create an output image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Create list of y-points to draw
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] ) 
 
    # Create a list of x-points from the y-points and the polynomial fits
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2] 
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    # plt.imshow(result)
    # plt.show()

    return result


# Use the best-fit lines in pixel space to get best-fit lines in real-world dimensions
# This function is based on code from the lessons.
def get_radii_of_curvature( img, left_fit, right_fit ):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Create list of y-points in pixel space
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
 
    # Create list of x-points in pixels space
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = img.shape[0]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature in world space (meters)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

