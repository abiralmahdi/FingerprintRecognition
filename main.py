import cv2
import os
from imageProcessing import resizeImage, featureExtractLaplace


# sample = cv2.imread("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/Altered/Altered-Hard/14__M_Right_middle_finger_CR.BMP")
sample = cv2.imread("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/TEST/100__M_testing.BMP")

bestScore = 0
filename = None
image = None
kp1, kp2, mp = None, None, None

#C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/Real
#C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/TEST/testing
counter = 0
for file in [file for file in os.listdir("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/Real")][:1000]:
    if counter%10==0:
        print(counter)
        print(file)
    counter+=1
    
    # fingerprint_image = cv2.imread("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/Real/"+file)
    fingerprint_image = cv2.imread("C:/Users/HP/Desktop/Course Materials/6. 5th Semester (Spring-24)/CSE468/Project/archive (1)/SOCOFing/Real/"+file)
    
    sift = cv2.SIFT_create()
    
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

    
    # algorithm:1 denotes the KD tree datastructure, number of trees=10
    matches = cv2.FlannBasedMatcher({'algorithm':1, 'trees':10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []
    
    for p, q in matches:
        if p.distance < 0.1*q.distance:
            match_points.append(p)
    
            
    keypoints = 0
    if len(keypoints_1) < len(keypoints_2):
        keypoints = len(keypoints_1)
    else:
        keypoints = len(keypoints_2)
    
    if len(match_points)/keypoints*100 > bestScore:
        bestScore = len(match_points)/keypoints*100
        filename = file
        image = fingerprint_image
        kp1, kp2, mp = keypoints_1, keypoints_2, match_points
    
print("BEST MATCH: "+str(filename))
print("SCORE: "+str(bestScore))

result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
resize = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()