import math
import cv2
import numpy as np 
import os.path
import random

"""
*** post process
"""

def calculate_cdf(histogram):
    """
    This method calculates the cumulative distribution function
    :param array histogram: The values of the histogram
    :return array normalized_cdf: The normalized cumulative distribution function
    """
    # Get the cumulative sum of the elements
    cdf = histogram.cumsum()
 
    # Normalize the cdf
    normalized_cdf = cdf / float(cdf.max())
 
    return normalized_cdf
 
def calculate_lookup(src_cdf, ref_cdf):
    """
    This method creates the lookup table
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return array lookup_table: The lookup table
    """
    lookup_table = np.zeros(256)
    lookup_val = 0
    for src_pixel_val in range(len(src_cdf)):
        lookup_val
        for ref_pixel_val in range(len(ref_cdf)):
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                lookup_val = ref_pixel_val
                break
        lookup_table[src_pixel_val] = lookup_val
    return lookup_table
 
def match_histograms(src_image, ref_image):
    """
    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image ref_image: The reference image
    :return image image_after_matching
    """
    # Split the images into the different color channels
    # b means blue, g means green and r means red
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)
 
    # Compute the b, g, and r histograms separately
    # The flatten() Numpy method returns a copy of the array c
    # collapsed into one dimension.
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0,256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0,256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0,256])    
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0,256])    
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0,256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0,256])
 
    # Compute the normalized cdf for the source and reference image
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)
 
    # Make a separate lookup table for each color
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)
 
    # Use the lookup function to transform the colors of the original
    # source image
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)
 
    # Put the image back together
    image_after_matching = cv2.merge([
        blue_after_transform, green_after_transform, red_after_transform])
    image_after_matching = cv2.convertScaleAbs(image_after_matching)
 
    return image_after_matching

def calc_max_rect(mask):
    """
    This method calculates the maximum rectangle region of the given binary mask.
    :param image mask: The binary mask
    :return list rect: 
    """
    row, col = mask.shape[:2]
    height = [0] * (col + 2)
    res = 0
    x1, y1, x2, y2 = 0, 0, 0, 0
    for i in range(row):
        stack = []
        for j in range(col + 2):
            if 1<=j<=col: 
                if mask[i,j-1] == 1:
                    height[j] += 1
                else:
                    height[j] = 0
            while stack and height[stack[-1]] > height[j]:
                cur = stack.pop()
                tmp = (j - stack[-1] - 1)* height[cur]
                if tmp > res:
                    res = tmp
                    x1 = stack[-1] # - 1 + 1
                    x2 = j - 1 - 1 
                    y1 = i - height[cur] + 1
                    y2 = i
            stack.append(j)
    return [x1, y1, x2, y2]

def get_rect(mask, scale=0.5, format="xyxy"):
    """
    :encapsulation of calc_max_rect.
    :param image mask: The binary mask
    :param integer scale: A factor to do trade-off between accuracy and efficiency
    :param string format: The coordinates format. ["xyxy", "xywh", "ctrwh"]
    :return list rect:
    """
    h, w = mask.shape[:2]
    h = int(h * scale)
    w = int(w * scale)
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    rect = calc_max_rect(mask_resized)
    rect = [int(i / scale) for i in rect]
    if format == "xyxy":
        return rect
    else:
        return [rect[0], rect[1], rect[2]-rect[0]+1, rect[3]-rect[1]+1]

def do_match_template(img, template, multi_scales=[(1,1)], dist=cv2.TM_CCOEFF_NORMED):
    ''' 
    :encapsulation of cv2.matchTemplate with multi-scales support
    :param image img: Input img
    :param image template: Input template
    :param list multi_scales: A list of scale ratios, [(w_ratio1, h_ratio1), (w_ratio2, h_ratio2), ...], [(1,1)] for the original scale
    :param integer dist: Distance metric for calculating the matching score
    :return list match_matrixes: A list of matching matrixes
    '''
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    templates = []
    h, w = template.shape[:2]
    for (w_ratio, h_ratio) in multi_scales:
        tmp = cv2.resize(template, (int(w * w_ratio), int(h * h_ratio)))
        templates.append(tmp)
    match_matrixes = []
    for t in templates:
        tmp = cv2.matchTemplate(img, t, dist)
        match_matrixes.append(tmp)
    return match_matrixes

def has_overlap(rect1, rect2):
    ''' 
    :This method determines whether there is an intersection between two rectangles
    :param list rect1: Rectangle 1
    :param list rect2: Rectangle 2
    :return boolean: 
    '''
    if not rect1 or not rect2:
        raise Exception("empty rectangle. rect1={}. rect2={}".format(rect1, rect2)) 
    if rect1[2] < rect2[0] or rect1[0] > rect2[2] or rect1[3] < rect2[1] or rect1[1] > rect2[3]:
        return False
    return True

def find_rects(fn, frames, masks):
    ''' 
    :This method finds a suitable region to do advertising placement. 
    :param string fn: Video file name (extention excluded)
    :param list frames: A list contains all frames of the video clip.
    :param list masks: A list contains all masks of the video clip.
    :return list anchors: Markers of the clip. i.e., [[[marker_1],[marker_2],...], [[marker_1],[marker_2],...]], ...], where marker_i = [x1,y1,x2,y2]
    :return list rects: Final advertising placement region of the clip. i.e., [rect_1, rect_2, ...], where rect_i = [x1,y1,x2,y2]
    '''
    rects = []

    # first step: detecting the anchors
    anchor_dir = os.path.join("anchors", fn)
    anchors = []
    if os.path.exists(anchor_dir):
        multi_anchor = []
        for anc_name in os.listdir("anchors/"+fn):
            anchor = cv2.imread("anchors/"+fn+"/"+anc_name)
            multi_anchor.append(anchor)
        threshold = 0.75
        for i, frame in enumerate(frames):
            if masks[i] is None:
                anchors.append([None] * len(multi_anchor))
                continue
            anchors_ = []
            for anchor in multi_anchor:
                h, w = anchor.shape[:2]
                match_matrix = do_match_template(frame, anchor)
                tmp = []
                # match_matrix = simple_nms(match_matrix)
                # loc = np.where(match_matrix[0] >= threshold)
                # for pt in zip(*loc[::-1]):
                #     tmp.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
                loc = np.unravel_index(np.argmax(match_matrix[0]), match_matrix[0].shape)
                if match_matrix[0][loc[0], loc[1]] > threshold:
                    tmp.append([loc[1], loc[0], loc[1]+w, loc[0]+h])
                if not tmp:
                    anchors_.append(None)
                else:
                    anchors_.append(list(np.mean(tmp, 0)))
            anchors.append(anchors_)
        print("anchor localization done.")
    else:
        raise Exception("Anchor folder of video {} not found".format(fn))

    # second step: calculating the maximum rectangle region of each mask 
    # These rects are used to determine the initial position and size of advertising placement region.
    min_area, max_area = 0, 0 # maintaining the maximum and minimum rectangle areas of the video clip
    min_area_idx, max_area_idx = 0, 0 # corresponding frame index

    for i, mask in enumerate(masks):
        if mask is None:
            continue
        elif i > 90 or i > len(masks): # we only use the first 90 frames (3s if fps is 30)
            break
        else:
            tmp = []
            for j in range(mask.shape[0]): # the mask may contain several detection results.
                m = mask[j, :, :, :]
                rect = get_rect(m)
                tmp.append(rect)

            if rects == []:
                # we choose the rect with bigger area as our initial rect if there are multiple rects in this frame
                tmp = np.array(tmp)
                area = (tmp[:,2] - tmp[:, 0]) * (tmp[:,3] - tmp[:,1])
                max_area_idx = np.argmax(area)
                max_area, min_area = area[max_area_idx], area[max_area_idx]
                rects.append(tmp[max_area_idx].tolist())
            else:
                max_overlap = 0 
                max_idx = 0

                for j in range(len(tmp)):
                    r = tmp[j]
                    rect = rects[0] # compare with the initial rect
                    if has_overlap(r, rect) is False:
                        overlap = 0
                        continue
                    w = min(r[2], rect[2]) - max(r[0], rect[0])
                    h = min(r[3], rect[3]) - max(r[1], rect[1])
                    overlap = w*h
                    if overlap > max_overlap:
                        max_overlap = overlap
                        max_idx = j
                if max_overlap != 0:
                    r = tmp[max_idx]
                    area = (r[2] - r[0]) * (r[3] - r[1])
                    if area > max_area:
                        max_area = area
                        max_area_idx = i
                    if area < min_area:
                        min_area = area
                        min_area_idx = i
                    rects.append(r)
    print("maximum rectangle calculation done.")

    
    # third step: combine the anchor with detected rect to obtain final rect.
    shared_rect = rects[0] # the suitable region 
    
    for i in range(1, len(rects)):
        r = rects[i]
        area = (r[2]-r[0])*(r[3]-r[1])
        if has_overlap(r, shared_rect): # and area != min_area and area != max_area:
            [x1, y1, x2, y2] = shared_rect
            shared_rect = [max(r[0], x1), max(r[1], y1), min(r[2], x2), min(r[3], y2)]
                
    #rects = np.array(rects)
    #shared_rect = [rects[:, 0].max(0), rects[:,1].max(0), rects[:,2].min(0), rects[:,3].min(0)]
    #shared_rect = [rects[:, 0].mean(0), rects[:,1].mean(0), rects[:,2].mean(0), rects[:,3].mean(0)]

    w, h = shared_rect[2] - shared_rect[0] + 1, shared_rect[3] - shared_rect[1] + 1

    rects = []
    dist_between_anchors = [] # distance between anchors, which is later used to adjust the rect size.
    for i, multi_anchor in enumerate(anchors):
        tmp = []
        for j, anchor in enumerate(multi_anchor):
            if j == len(multi_anchor) - 1 :
                next_anchor = multi_anchor[0]
            else:
                next_anchor = multi_anchor[j+1]
            if anchor is None or next_anchor is None:
                tmp.append(None)
            else:
                tmp.append(abs(anchor[0] - next_anchor[0]))
        dist_between_anchors.append(tmp)

    dist = [] # distance between the shared_rect and anchors
    for i in range(len(masks)):
        mask = masks[i]
        bad_anchor = True # if none of the anchors are found in this frame, which means camera view changes a lot, we then skip this frame
        multi_anchor = anchors[i]
        for anchor in multi_anchor:
            if anchor is not None:
                bad_anchor = False 
                break
        if mask is None or bad_anchor is True: # mask is None means this frame does not belong to the video clip.
            rects.append(None)
        else:
            if not dist:
                for anchor in multi_anchor:
                    dist.append([shared_rect[0] - anchor[0], shared_rect[1] - anchor[1]]) # the initial relative offset
            
            w_scaled, h_scaled = w, h
            if i > 1:
                sclae = 1
                tmp = []
                for (dist1, dist2) in zip(dist_between_anchors[i], dist_between_anchors[0]):
                    if dist1 is not None and dist2 is not None:
                        if dist1 == 0 or dist2 == 0:
                            tmp.append(1)
                        else:
                            tmp.append(float(dist1) / float(dist2))
                if tmp:
                    scale = sum(tmp) / len(tmp)
                scale = 1 # debug, remove later
                w_scaled, h_scaled = int(w * scale), int(h * scale)

            tmp = []
            # final rect = anchor position + relative offset ( + width / height)  
            for j, anchor in enumerate(multi_anchor):
                if anchor is None:
                    continue                
                tmp.append([anchor[0]+dist[j][0], anchor[1]+dist[j][1], anchor[0]+dist[j][0]+w_scaled, anchor[1]+dist[j][1]+h_scaled])
            tmp = np.array(tmp)
            rects.append([tmp[:,0].mean(0), tmp[:,1].mean(0), tmp[:,2].mean(0), tmp[:,3].mean(0)])
    print("final rect generation done.")
    return anchors, rects

def add_ad(img, rect, anchors=None, ad_path="ad/3.png"):
    ''' 
    :This method puts the ad image into the detected rectanle region.
    :param image img: Video frame
    :param list rect: Where to put ad image on
    :param list anchors: If anchors are not None, we draw it on the frame
    :return string ad_path: File path of the ad image
    :return image img: 
    '''
    if rect is None:
        return img
    if not os.path.exists(ad_path):
        raise Exception("File not found: {}".format(ad_path))
    ad = cv2.imread(ad_path)

    [x1, y1, x2, y2] = list(map(int, rect))
    W = x2 - x1 + 1
    H = y2 - y1 + 1
    if W < 0 or H < 0:
        raise Exception("Bad rect: {}".format(rect))
 
    ad = cv2.resize(ad,(W,H))
    # shift the ad img accoording to the camera movements (i.e., rect exceeds the image)
    x11,y11,x22,y22 = 0,0,W-1,H-1
    
    if x1 > img.shape[1] - 1 or x2 < 0 or y1 > img.shape[0] - 1 or y2 < 0:
        print("target lost.")
    else:
        if x1 < 0:
            x11 += abs(x1)
        if y1 < 0:
            y11 += abs(y1)
        if x2 > img.shape[1] - 1:
            x22 -= x2 - img.shape[1] + 1
        if y2 > img.shape[0] - 1:
            y22 -= y2 - img.shape[0] + 1

        x1 =max(int(x1),0)
        x2 =min(int(x2),img.shape[1])
        y1 =max(int(y1),0)
        y2 =min(int(y2),img.shape[0])

        img_ref = img.copy()
        img[y1:y2+1,x1:x2+1,:] = ad[y11:y22+1, x11:x22+1, :]
        img = match_histograms(img, img_ref)    

    if anchors is None:
        return img
    for anchor in anchors:
        if anchor is not None:
            anchor = list(map(int, anchor))
            img = cv2.rectangle(img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), (0,255,0), 2)
    return img
        
