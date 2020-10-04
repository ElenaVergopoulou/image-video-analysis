import numpy as np
import cv2
import skimage.util as ski
from skimage import filters
from skimage.morphology import disk
from skimage import feature
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

fourcc = cv2.VideoWriter_fourcc(*'MP4V')


#Question 1

# video.mp4: input video
cap = cv2.VideoCapture("VIRAT_S_000201_02_000590_000623.mp4")
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

#we set the new height and width to get half the resolytion
height = int(height//2)
width=int(width//2)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')

Resized = cv2.VideoWriter('Resized.mp4',fourcc, 20.0, (width,height))


while (cap.isOpened()):
  ret, frame = cap.read()

  if ret==False:
  	break
  #resize each frame
  frame=cv2.resize(frame, (width, height))
  Resized.write(frame)

  cv2.imshow("Resized frames", frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

#release all opened videos and close all popped windows
cap.release()
Resized.release()
cv2.destroyAllWindows()

#Question 2

Resized = cv2.VideoCapture('Resized.mp4')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')

#save 3 videos for all the different colorspaces
Gray = cv2.VideoWriter('Gray.mp4',fourcc, 20.0, (width,height),isColor=False)
HSV = cv2.VideoWriter('HSV.mp4',fourcc, 20.0, (width,height))
LAB = cv2.VideoWriter('LAB.mp4',fourcc, 20.0, (width,height))

flag=0
while (Resized.isOpened()):
  ret, frame = Resized.read()

  if ret==False:
  	break

  grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  labFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

  #screenshot only for the first frame for report
  if flag==0:
  	cv2.imwrite('gray_colorspace.jpg', grayFrame)
  	cv2.imwrite('hsv_colorspace.jpg', hsvFrame)
  	cv2.imwrite('lab_colorspace.jpg', labFrame)
  flag=1

  Gray.write(grayFrame)
  HSV.write(hsvFrame)
  LAB.write(labFrame)

  cv2.imshow("Gray frame", grayFrame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
      break

Resized.release()
Gray.release()
HSV.release()
LAB.release()
cv2.destroyAllWindows()
Gray = cv2.VideoCapture('Gray.mp4')

#Question 3

#different combinations tested that can be seen in the report
#maxCorners, qualityLevel, minDistance = 50, 0.01, 10
#maxCorners, qualityLevel, minDistance = 50, 0.01, 10
#maxCorners, qualityLevel, minDistance = 100, 0.01, 10
#maxCorners, qualityLevel, minDistance = 200, 0.01, 5
#maxCorners, qualityLevel, minDistance = 200, 0.01, 10
#maxCorners, qualityLevel, minDistance = 200, 0.005, 10
#maxCorners, qualityLevel, minDistance = 200, 0.01, 20
#maxCorners, qualityLevel, minDistance = 200, 0.1, 10
maxCorners, qualityLevel, minDistance = 300, 0.01, 10

#Harris

#first frame from resized
f = cv2.VideoCapture('Resized.mp4')
rval, image = f.read()
f.release()

#turn it to gray for the algorithm
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#goodFeaturesToTrack is called like
#goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[)

corners = cv2.goodFeaturesToTrack(gray,maxCorners, qualityLevel, minDistance, useHarrisDetector=True)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(image,(x,y),3,255,-1)
cv2.imshow('Harris Corner Detector', image)
cv2.imwrite('first_frame_harris_' + str(maxCorners) + '_' + str(qualityLevel) + '_' + str(minDistance)+'.jpg', image)


#Shi-Tomashi
f = cv2.VideoCapture('Resized.mp4')
rval, image = f.read()
f.release()
# convert corners values to integer 
# So that we will be able to draw circles on them 

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray,maxCorners, qualityLevel, minDistance)
corners = np.int0(corners)

# draw color circles on all corners 
for i in corners:
    x,y = i.ravel()
    cv2.circle(image,(x,y),3,255,-1)


# resulting image 
cv2.imshow('Shi-Tomashi Detector', image)
cv2.imwrite('first_frame_shitomashi_'  + str(maxCorners) + '_' + str(qualityLevel) + '_' + str(minDistance)+'.jpg', image)


#Question 4

#function for all Lucas-Kanade

def LucasKanade(name, feature_params, lk_params, method, pro=False, noise=False, denoise=False):	
	#shitomasi
	Algorithm = cv2.VideoWriter(name + '_' + method + '.mp4',fourcc, 20.0, (width,height))

	cap = cv2.VideoCapture("Resized.mp4")
	ret, first_frame = cap.read()
	# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
	prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

	if noise:
		SPNoise =  ski.random_noise(prev_gray,mode = 's&p', seed = 0 ,amount = SP_Amount(0))
		prev_gray= np.array(255*SPNoise, dtype = 'uint8')

	if denoise:
		SPMedian= filters.rank.median(prev_gray, np.ones((5,5)))
		prev_gray = np.array(255*SPMedian, dtype = 'uint8')

	# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
	if(method=='Shi-Tomasi'):
		prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
		color = (0, 0, 255)
	else:
		prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params, useHarrisDetector=True)
		color = (255, 0, 0)

	#save first frame
	ff=prev
	# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
	mask = np.zeros_like(first_frame)
	num=0
	k=0
	take_screenshot_every=400
	screenshots = []
	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret==False:
			break
		fr=1
		if(pro):
			fr=10
		if num%fr==0:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			if noise:
				SPNoise =  ski.random_noise(gray,mode = 's&p', seed = 0 ,amount = SP_Amount(0))
				gray= np.array(255*SPNoise, dtype = 'uint8')

			if denoise:
				SPMedian= filters.rank.median(gray, np.ones((5,5)))
				gray = np.array(255*SPMedian, dtype = 'uint8')

			# Calculates sparse optical flow by Lucas-Kanade method
			next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
			# Selects good feature points for previous position
			good_old = prev[status == 1]
			# Selects good feature points for next position
			good_new = next[status == 1]
			# Draws the optical flow tracks
			for i, (new, old) in enumerate(zip(good_new, good_old)):
			    # a, b = coordinates of new point
			    a, b = new.ravel()
			    # c, d = coordinates of old point
			    c, d = old.ravel()

			    flag= True
			    if (math.sqrt((a-c)**2+(b-d)**2)<1):
			    	flag = False

			    if (flag):
			    	# Draws line between new and old position with green color and 2 thickness
				    mask = cv2.line(mask, (a, b), (c, d), color, 2)
				    # Draws filled circle (thickness of -1) at new position with green color and radius of 3
				    frame = cv2.circle(frame, (a, b), 3, color, -1)
				    k=k+1

			# Overlays the optical flow tracks on the original frame
			output = cv2.add(frame, mask)
			# Updates previous frame
			prev_gray = gray.copy()
			# Updates previous good feature points
			prev = good_new.reshape(-1, 1, 2)
			Algorithm.write(output)
			# Opens a new window and displays the output frame
			cv2.imshow(name + ' ' + method, output)

			if num%take_screenshot_every==0:
				cv2.imwrite(name + '_' + method + '_' + str(num) + '.jpg', output)
				#screenshots.append(output)

		num=num+1
		# Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
		if cv2.waitKey(10) & 0xFF == ord('q'):
		    break
	# The following frees up resources and closes all windows
	cap.release()
	Algorithm.release()

	cv2.destroyAllWindows()


# Parameters for corner detection (different combinations tested)

#feature_params = dict(maxCorners = 200, qualityLevel = 0.01, minDistance = 10)
feature_params = dict(maxCorners = 300, qualityLevel = 0.01, minDistance = 10)

# Parameters for Lucas-Kanade optical flow (all different combinations tested)

#lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#lk_params = dict(winSize = (5,5), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
lk_params = dict(winSize = (30,30), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#lk_params = dict(winSize = (30,30), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#lk_params = dict(winSize = (30,30), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#lk_params = dict(winSize = (30,30), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2, 0.1))
#lk_params = dict(winSize = (30,30), maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

#Actually Question 4

LucasKanade('Lukas-Kanade', feature_params, lk_params, 'Shi-Tomasi')
LucasKanade('Lukas-Kanade', feature_params, lk_params, 'Harris')


#Question 5

LucasKanade('Lukas-Kanade without stable points', feature_params, lk_params, 'Shi-Tomasi', True)
LucasKanade('Lukas-Kanade without stable points', feature_params, lk_params, 'Harris', True)

#Question 6

#function from question one used for the noise inputs
def SP_Amount(x):
  amount = x/90 + 0.3
  return amount

LucasKanade('Lukas-Kanade with noise', feature_params, lk_params, 'Shi-Tomasi', True, True)
LucasKanade('Lukas-Kanade with noise', feature_params, lk_params, 'Harris', True, True)


#Question 7

LucasKanade('Lukas-Kanade after filter', feature_params, lk_params, 'Shi-Tomasi', True, True, True)
LucasKanade('Lukas-Kanade after filter', feature_params, lk_params, 'Harris', True, True, True)
