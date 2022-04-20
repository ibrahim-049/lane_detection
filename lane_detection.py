import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip





def abs_sobel_thresh(gray, thresh=(0, 255)):

    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    sobel_mask = binary_output
    
    return sobel_mask


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    dir_mask = binary_output
    return dir_mask

def plotImg(original, titleOriginal, processed_image, processed_title, cmap='gray'):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 10))
    ax1.set_title(titleOriginal, fontsize=30)
    ax1.imshow(original, cmap='gray')
    ax2.set_title(processed_title, fontsize=30)
    ax2.imshow(processed_image, cmap='gray')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def combine_thresh(img, s_thresh=(100, 255), l_thresh=(120, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    
    grad_x = abs_sobel_thresh(gray, thresh=(10, 200))
    dir_binary = dir_threshold(gray, thresh=(np.pi/6, np.pi/2))
    combined = ((grad_x == 1) & (dir_binary == 1))
    color_threshold = 150
    R = img[:,:,0]
    G = img[:,:,1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)
    

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    s_channel = hls[:,:,2] 
    l_channel = hls[:,:,1]
    s_condition = (s_channel > s_thresh[0]) & (s_channel <= s_thresh[1]) 
    l_condition = (l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])
    color_combined[(r_g_condition & l_condition) & (s_condition | combined)] = 1
    mask = np.zeros_like(color_combined)
    region_of_interest_vertices = np.array([ [170,height-1], [width/2, int(5.7/10*height)], [1200*(img.shape[1]/1280)-1, height-1]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    color_combined = cv2.bitwise_and(color_combined, mask)

    return color_combined

def get_warp_points(image):
    
    scale_factor_column = image.shape[1] / 1280
    scale_factor_row = image.shape[0] / 720
    middle_point = image.shape[1] / 2

    corners = np.float32([[middle_point - (scale_factor_column*387), image.shape[0] - (scale_factor_row*23)], 
                          [middle_point - (scale_factor_column*115), image.shape[0] - (scale_factor_row*230)], 
                          [middle_point + (scale_factor_column*130), image.shape[0] - (scale_factor_row*230)], 
                          [middle_point + (scale_factor_column*421), image.shape[0] - (scale_factor_row*30)]])

    top_left = np.array([corners[0, 0], 0])
    top_right = np.array([corners[3, 0], 0])
    offset = [50, 0]
    

    img_size = (image.shape[1], image.shape[0])
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32([corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])
    
    return src_points, dst_points

def warp_image_to_birdseye_view(img, src, dst, image_size):
    perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, perspectiveTransform, image_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, Minv

def warp_image_to_birdseye_view_gray(image):
    src, dst = get_warp_points(image)
    img = combine_thresh(image,s_thresh=(100, 255), l_thresh=(120, 255))
    warped, Minv = warp_image_to_birdseye_view(img, src, dst, (image.shape[1], image.shape[0]))
    
    return img, warped, Minv

temp_array_left = []
temp_array_right = []


def fitlines(warped, nwindows=15, margin=100, minpix = 50 ):
    
    global temp_array_left
    global temp_array_right
    
    temp_array_left = []
    temp_array_right = []
    
    histogram = np.sum(warped, axis=0)
    out_img = np.dstack((warped, warped, warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(warped.shape[0]/nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    
    for window in range(nwindows):

        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            temp_array_left.append(window)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            temp_array_right.append(window)
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    

    if len(leftx) == 0:
        left_fit =[]
    else:
        left_fit = np.polyfit(lefty, leftx, 2)
    
    if len(rightx) == 0:
        right_fit =[]
    else:
        right_fit = np.polyfit(righty, rightx, 2)

        
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
   
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    

    return left_fit, right_fit,out_img, lefty, leftx, righty, rightx, ploty
     

def lane_curvatures(img, lefty, leftx, righty, rightx, ploty):
    
    
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/730 
    
    y_eval = np.max(ploty) 

    left_curverad = 0
    right_curverad = 0
    center = 0

    if len(leftx) != 0 and len(lefty) != 0 and len(righty) != 0 and len(righty) != 0:

        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center = (car_pos - lane_center_position) * xm_per_pix / 10


    return center,left_curverad, right_curverad

old_lane = [[[]]]
def draw_lane(img, warped, left_fit, right_fit, ploty, center, left_curverad, right_curverad, M_inv):
   
    
    global old_lane
    global temp_array_left
    global temp_array_right
    
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
  

    if ( (len(temp_array_left) > 4 and len(temp_array_right) > 4) and (temp_array_left[-1] > 1 and temp_array_left[0] < 8) and (temp_array_right[-1] > 1 and temp_array_right[0] < 7)):
        
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp,np.int_([pts_left]),False,(255,0,0),30)
        cv2.polylines(color_warp,np.int_([pts_right]),False,(0,0,255),30)

        newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0])) 

        TextL = "Left  Curvature: " + str(int(left_curverad)) + " m"
        TextR = "Right Curvature: " + str(int(right_curverad))+ " m"
        if (center > 0):
            TextC = "Vehicle is " + str(round(float(center),3))+ "m right of center"
        elif (center < 0):
            TextC = "Vehicle is " + str(round(float(center),3))+ "m left of center"    
        else:
            TextC = "Vehicle is exactly on center"
            
        fontScale=1.5
        thickness=3
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
         
        cv2.putText(newwarp, TextL, (100,60), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        cv2.putText(newwarp, TextR, (100,110), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        cv2.putText(newwarp, TextC, (100,170), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        old_lane = newwarp.copy()
     
        
    else:
        newwarp = old_lane.copy()    
        src,dst = get_warp_points(newwarp)
        color_warp, _ = warp_image_to_birdseye_view(newwarp, src, dst, (newwarp.shape[1], newwarp.shape[0]))
        
  

    
    
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result,color_warp

def createVideo(input_clip, output_clip="./output.mp4", debug="0"):
    
       
    if(debug == "0"):
 
        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(software_pipeline_v1) 
        out_clip.write_videofile(output_clip, audio=False)

    else:

        clip1 = VideoFileClip(input_clip)
        out_clip = clip1.fl_image(software_pipeline_v2) 
        out_clip.write_videofile(output_clip, audio=False)
        
    return output_clip  


def software_pipeline_v1(img):
   
    thresholded, binary_warped, M_inv = warp_image_to_birdseye_view_gray(img)

    left_fit, right_fit,out_img, lefty, leftx, righty, rightx, ploty = fitlines(binary_warped,15)    
    

    center,left_curverad, right_curverad = lane_curvatures(img, lefty, leftx, righty, rightx, ploty)
    
    processed_img,color_warp = draw_lane(img, binary_warped, left_fit, right_fit, ploty, center,left_curverad, right_curverad, M_inv)
    

    return processed_img


def software_pipeline_v2(img):

    thresholded, binary_warped, M_inv = warp_image_to_birdseye_view_gray(img)
    
   
    left_fit, right_fit,out_img, lefty, leftx, righty, rightx, ploty = fitlines(binary_warped,15)    
    

    center,left_curverad, right_curverad = lane_curvatures(img, lefty, leftx, righty, rightx, ploty)
    

    processed_lane,color_warp = draw_lane(img, binary_warped, left_fit, right_fit, ploty, center,left_curverad, right_curverad, M_inv)
    

    new_threshold = np.zeros_like(out_img)
    new_threshold[ (thresholded == 1)] = [255,255,255]
    added_img = np.concatenate((color_warp,new_threshold),axis = 0)
    added_img = cv2.resize(added_img,(int(0.55*img.shape[1]),img.shape[0]))
    processed_img = np.concatenate((processed_lane,added_img),axis = 1)
    new_warped = np.zeros_like(out_img)
    new_warped[ (binary_warped == 1)] = [255,255,255]
    added_img = np.concatenate((new_warped,out_img),axis = 1)
    added_img = cv2.resize(added_img,(processed_img.shape[1],int(0.6*processed_img.shape[0])))
    processed_img = np.concatenate((processed_img,added_img),axis = 0)    
    

    return processed_img


def main():
    createVideo(sys.argv[1], sys.argv[2], sys.argv[3]) 

if __name__ == "__main__":
    main()