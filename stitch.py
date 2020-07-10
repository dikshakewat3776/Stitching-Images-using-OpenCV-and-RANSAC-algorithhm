import numpy as np
import cv2
import argparse
import glob
import random as rand
import os
from scipy.spatial.distance import cdist
from scipy.spatial import distance


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Image directory')
    parser.add_argument('--Image_Directory', type=str, default='./data')
    parser.add_argument('--Result_Directory', type=str, default='./results')
    args = parser.parse_args()
    return args

# Reading the images.
def read_images(imageDir, show=False):
    images = [cv2.imread(file) for file in glob.glob(imageDir)]
    colorImages = [cv2.imread(file,1) for file in glob.glob(imageDir)]
    img_dict = {}
    img_gray_dict = {}
    i = 0   
    for image in images:
        if show:
            show_images(image)
        key = f"img{i}"   
        value = images[i]
        img_dict[key] = value
        gray_key = f"gray_img{i}"
        value_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        img_gray_dict[gray_key] = value_gray
        i += 1
    print('Reading {} images successfully'.format(i))
    return img_gray_dict


def show_images(images,delay=1000):
    """Shows all images.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', images)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

#Finding Key points
def find_image_key_points(img_gray_dict, show_keypoints= False):
    sift = cv2.xfeatures2d.SIFT_create()#SIFT (Scale-Invariant Feature Transform)
    i = 0
    key_points = []
    descriptors = []
    if img_gray_dict:
        for i in range(len(img_gray_dict)):
            key_pt = f"kp{i}"
            desc = f"desc{i}"
            gray_image_dict = img_gray_dict[f"gray_img{i}"]
            key_pt, desc = sift.detectAndCompute(gray_image_dict, None)
            key_points.append(key_pt)
            descriptors.append(desc)
            if show_keypoints:
                cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('image_key_points', cv2.drawKeypoints(img_gray_dict[f"gray_img{i}"], key_points[i], None))
                cv2.waitKey(delay=1000)
    else:
        image_key_points = None
        image_descriptors = None
    image_key_points = key_points
    image_descriptors = descriptors
    print('Number of Key points is {}'.format(len(image_key_points)))
    print('Number of Descriptors is {}'.format(len(descriptors)))
    return image_key_points,image_descriptors

#Matching key points using squared euclidean distance
def matching_key_points(all_image_gray_dict,all_image_key_points,all_image_descriptors):
    num_loops = range(len(all_image_gray_dict))
    for i in num_loops:
        if i == len(all_image_gray_dict)-1:
            break
        distance = get_distance_sqeuclidian(all_image_key_points[i],all_image_key_points[i+1],all_image_descriptors[i],all_image_descriptors[i+1])
    print("Matching key points of {} images".format(len(all_image_gray_dict)))
    # print(distance)
    return distance
    
#squared euclidean distance between two images
def get_distance_sqeuclidian(kps1, kps2, desc1, desc2):
    pairwiseDistances = cdist(desc1, desc2, 'sqeuclidean')
    threshold   = 10000
    points_in_img1 = np.where(pairwiseDistances < threshold)[0]
    points_in_img2 = np.where(pairwiseDistances < threshold)[1]
    coordinates_in_img1 = np.array([kps1[point].pt for point in points_in_img1])
    coordinates_in_img2 = np.array([kps2[point].pt for point in points_in_img2])
    return np.concatenate((coordinates_in_img1,coordinates_in_img2), axis=1)


#computing homography matrix and implementing ransac algorithm
#ransac algorithm
def ransac_algo(matchingPoints,totalIteration):
    
    # Ransac parameters
    highest_inlier_count = 0
    best_H = []
    
    # Loop parameters
    counter = 0
    while counter < totalIteration:
        counter = counter + 1
        # Select 4 points randomly
        secure_random  = rand.SystemRandom()
        
        matachingPair1 = secure_random.choice(matchingPoints)
        matachingPair2 = secure_random.choice(matchingPoints)
        matachingPair3 = secure_random.choice(matchingPoints)
        matachingPair4 = secure_random.choice(matchingPoints)
        
        fourMatchingPairs=np.concatenate(([matachingPair1],[matachingPair2],[matachingPair3],[matachingPair4]),axis=0)
        
        # Finding homography matrix for this 4 matching pairs
        # H = get_homography(fourMatchingPairs)

        points_in_image_1 = np.float32(fourMatchingPairs[:,0:2])
        points_in_image_2 = np.float32(fourMatchingPairs[:,2:4])
        
        H = cv2.getPerspectiveTransform(points_in_image_1, points_in_image_2)
        
        rank_H = np.linalg.matrix_rank(H)
        
        # Avoid degenrate H
        if rank_H < 3:
            continue
        
        # Calculate error for each point using the current homographic matrix H
        total_points = len(matchingPoints)
        
        points_img1 = np.concatenate( (matchingPoints[:, 0:2], np.ones((total_points, 1))), axis=1)
        points_img2 = matchingPoints[:, 2:4]
        
        correspondingPoints = np.zeros((total_points, 2))
        
        for i in range(total_points):
            t = np.matmul(H, points_img1[i])
            correspondingPoints[i] = (t/t[2])[0:2]

        error_for_every_point = np.linalg.norm(points_img2 - correspondingPoints, axis=1) ** 2

        inlier_indices = np.where(error_for_every_point < 0.5)[0]
        inliers        = matchingPoints[inlier_indices]
    
        curr_inlier_count = len(inliers)
      
        if curr_inlier_count > highest_inlier_count:
            highest_inlier_count = curr_inlier_count
            best_H = H.copy()

    return best_H

def stitch_images(all_image_gray_dict, matched_key_points):
    num_loops = range(len(all_image_gray_dict))
    for i in num_loops:
        if i == len(all_image_gray_dict)-1:
            break
        stitch= ransac_algo(matched_key_points,7000)
    print("Stitching {} images".format(len(all_image_gray_dict)))
    # print(distance)
    return stitch
    

#main function
def main():
    args = parse_args()
    imageDir = args.Image_Directory + '/*.jpg'
    all_image_gray_dict = read_images(imageDir = imageDir, show = True)
    all_image_key_points,all_image_descriptors = find_image_key_points(all_image_gray_dict, show_keypoints = True)
    matched_key_points = matching_key_points(all_image_gray_dict,all_image_key_points,all_image_descriptors)
    all_stitch= stitch_images(all_image_gray_dict, matched_key_points)
    # stitch= ransac_algo(matched_key_points,7000)
    colorImages = [cv2.imread(file,1) for file in glob.glob(imageDir)]
    result = cv2.warpPerspective(colorImages[0],all_stitch,(int(colorImages[0].shape[1] + colorImages[1].shape[1]*0.8),int(colorImages[0].shape[0] + colorImages[1].shape[0]*0.4) ))
    result[0:colorImages[1].shape[0], 0:colorImages[1].shape[1]] = colorImages[1]   
    resultDir =  args.Result_Directory + '/result_image.jpg'
    cv2.imwrite(resultDir, result)
    show_images(result)

if __name__ == "__main__":
    main()
