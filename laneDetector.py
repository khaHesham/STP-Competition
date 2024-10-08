
import numpy as np
import cv2
import time



class DetectorAPI:
    def __init__(self):
        self.out_examples = 0
        self.MOV_AVG_LENGTH = 5
        self.smoothed_angle = 0

    def color(self,inpImage): 
        """i will write documentation later"""
        # Apply HLS color filtering to filter out white lane lines
        hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 160, 10])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(inpImage, lower_white, upper_white)
        hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)
        
        # Convert image to grayscale, apply threshold, blur & extract edges
        gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh,(3, 3), 11)
        canny = cv2.Canny(blur, 40, 60)
        return canny

    def sobel_binary(self,img, sobel_kernel=7, mag_thresh=(3, 255), s_thresh=(170, 255)):
        """i will write documentation later"""
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # Binary matrixes creation
        sobel_binary = np.zeros(shape=gray.shape, dtype=bool)
        s_binary = sobel_binary
        combined_binary = s_binary.astype(np.float32)
        # Sobel Transform
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = 0 
        #cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobel_abs = np.abs(sobelx**2 + sobely**2)
        sobel_abs = np.uint8(255 * sobel_abs / np.max(sobel_abs))
        sobel_binary[(sobel_abs > mag_thresh[0]) & (sobel_abs <= mag_thresh[1])] = 1
        # Threshold color channel
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Combine the two binary thresholds
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
        combined_binary = np.uint8(255 * combined_binary / np.max(combined_binary))
        return combined_binary

    def region_of_interest(self,img):
        mask = np.zeros_like(img)
        imshape=img.shape
        vertices = np.array([[(150,imshape[0]),(590, 440), (680, 440), (imshape[1]-20,imshape[0])]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def warp(self,img, src, dst):
        """i will write documentation later"""
        src = np.float32([src])
        dst = np.float32([dst])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                dsize=img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)


    def sliding_windown(self,img_w):
        """i will write documentation later"""
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img_w, img_w, img_w)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = int(img_w.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        # ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        #
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        return left_fit, right_fit


    def fit_from_lines(self,left_fit, right_fit, img_w):
        """i will write documentation later"""
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
        (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit


    def draw_lines(self,img, img_w, left_fit, right_fit, perspective):
        """i will write documentation later"""
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img_w).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        #color_warp_center = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        #cv2.fillPoly(color_warp_center, np.int_([pts]), (0, 255, 0))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.warp(color_warp, perspective[1], perspective[0])
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.2, 0)

        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))
        cv2.polylines(color_warp_lines, np.int_([pts_right]), isClosed=False, color=(255, 255, 255), thickness=25)
        cv2.polylines(color_warp_lines, np.int_([pts_left]), isClosed=False, color=(0, 255, 255), thickness=25)
        newwarp_lines = self.warp(color_warp_lines, perspective[1], perspective[0])

        result = cv2.addWeighted(result, 1, newwarp_lines, 1, 0)

        # ----- Radius Calculation ------ #

        img_height = img.shape[0]
        y_eval = img_height

        ym_per_pix = 30 / 720.  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        ploty = np.linspace(0, img_height - 1, img_height)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])

        right_curverad = (
                            (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        radius1 = round((float(left_curverad) + float(right_curverad))/2.,2)
        
        if left_fitx[0] - left_fitx[-1] > 60:
            curve_direction = 'Left Curve'
            radius=-5729.57795/radius1
        elif left_fitx[-1] - left_fitx[0] > 60:
            curve_direction = 'Right Curve'
            radius=5729.57795/radius1
        else:
            curve_direction = 'Straight'
            radius=5729.57795/radius1

        # ----- Off Center Calculation ------ #

        lane_width = (right_fit[2] - left_fit[2]) * xm_per_pix
        center = (right_fit[2] - left_fit[2]) / 2
        off_left = (center - left_fit[2]) * xm_per_pix
        off_right = -(right_fit[2] - center) * xm_per_pix
        off_center = round((center - img.shape[0] / 2.) * xm_per_pix,2)

        # --- Print text on screen ------ #
        #if radius < 5000.0:
        text = "Angle = %s [degrees]\noffcenter = %s [m]" % (str(radius), str(off_center))
        #text = "radius = -- [m]\noffcenter = %s [m]" % (str(off_center))

        for i, line in enumerate(text.split('\n')):
            i = 550 + 20 * i
            cv2.putText(result, line, (0,i), cv2.FONT_HERSHEY_DUPLEX, 0.5,(255,255,255),1,cv2.LINE_AA)
        return result,radius

    def text(img,angle):
        """i will write documentation later"""
        mask = np.zeros_like(img)
        cv2.putText(img,str(round(angle)),(100,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)
        res = cv2.addWeighted(img,0.7,mask,0.3,0)
        return res

    def Run(self):
        """i will write documentation later"""
        cap = cv2.VideoCapture('LaneToSteeringChallenge.mp4')
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('LaneToSteeringChallenge.avi',fourcc, 20.0, (1280,720))
        #out.write(final)
        while(True):
            start=time.time()
            ret, frame = cap.read()
            if ret is True: 
                image = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
            else:
                print("Getting no frames")
                break
            #-------------------------Color & Gradient Threshold------------------------ 
            edges = self.color(image)
            #edges1 = canny(image)
            edges2=self.sobel_binary(image)
            
            A= cv2.addWeighted(edges2,0.7,edges,0.3,0)
            BW1=cv2.bitwise_and(A, edges2)
            img_b=self.region_of_interest(BW1)

        # ---------------------------- Perspective Transform --------------------------

            #src = [585, 457], [700, 457], [1110, img_b.shape[0]], [220, img_b.shape[0]]
            src = [480, 500], [800, 500], [img_b.shape[1]-50, img_b.shape[0]],  [150, img_b.shape[0]]

            line_dst_offset = 200
            #src = [595, 500], [685, 500],[1110,img_b.shape[0]],[220, img_b.shape[0]]
            dst = [src[3][0] + line_dst_offset, 0], \
                [src[2][0] - line_dst_offset, 0], \
                [src[2][0] - line_dst_offset, src[2][1]], \
                [src[3][0] + line_dst_offset, src[3][1]]
            
            img_w = self.warp(img_b, src, dst)
            try:
                left_fit, right_fit = self.fit_from_lines(left_fit, right_fit, img_w)
                mov_avg_left = np.append(mov_avg_left,np.array([left_fit]), axis=0)
                mov_avg_right = np.append(mov_avg_right,np.array([right_fit]), axis=0)
                
            except Exception:
                left_fit, right_fit = self.sliding_windown(img_w)
                mov_avg_left = np.array([left_fit])
                mov_avg_right = np.array([right_fit])

            left_fit = np.array([np.mean(mov_avg_left[::-1][:,0][0:self.MOV_AVG_LENGTH]),
                                np.mean(mov_avg_left[::-1][:,1][0:self.MOV_AVG_LENGTH]),
                                np.mean(mov_avg_left[::-1][:,2][0:self.MOV_AVG_LENGTH])])
            right_fit = np.array([np.mean(mov_avg_right[::-1][:,0][0:self.MOV_AVG_LENGTH]),
                                np.mean(mov_avg_right[::-1][:,1][0:self.MOV_AVG_LENGTH]),
                                np.mean(mov_avg_right[::-1][:,2][0:self.MOV_AVG_LENGTH])])
            if mov_avg_left.shape[0] > 1000:
                mov_avg_left = mov_avg_left[0:self.MOV_AVG_LENGTH]
            if mov_avg_right.shape[0] > 1000:
                mov_avg_right = mov_avg_right[0:self.MOV_AVG_LENGTH]
                
            final,degrees = self.draw_lines(frame, img_w, left_fit, right_fit, perspective=[src,dst])    
            self.smoothed_angle += 0.2 * pow(abs((degrees - self.smoothed_angle)), 2.0 / 3.0) * (degrees - self.smoothed_angle) / abs(degrees - self.smoothed_angle)
            # M = cv2.getRotationMatrix2D((scols/2,srows/2),-self.smoothed_angle,1)
            # dst = cv2.warpAffine(swheel,M,(scols,srows)) 
            # steer=text(dst,-self.smoothed_angle)
            # final[0:240, 0:240] = steer

            #out.write(final)
            cv2.imshow('front_view', frame)
            cv2.imshow('canny', edges)
            cv2.imshow('ROI', img_b)
            cv2.imshow('Sky_view', img_w)
            cv2.imshow('final', final)
            #cv2.imshow("steering wheel", dst)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def main():
    API = DetectorAPI()
    API.Run()


if __name__ == '__main__':
    main()
