# USAGE
# To read and write back out to video:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_01.mp4 \
#	--output output/output_01.avi
#
# To read from webcam and write back out to disk:
# python people_counter.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel \
#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from gluoncv import model_zoo, data, utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import mxnet as mx


def make_prediction(in_filename, out_filename, confidence_threshold=0.3, skip_frames=200, caffe_prototxt_file=None, model_file=None):
	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	print("[INFO] loading model...")
	# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

	# if a video path was not supplied, grab a reference to the webcam
	if not in_filename:
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] opening video file...")
		vs = cv2.VideoCapture(in_filename)

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	totalIn = 0
	totalOut = 0

	# start the frames per second throughput estimator
	fps = FPS().start()

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		frame = frame[1] if in_filename else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if in_filename is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		frame = imutils.resize(frame, width=500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if out_filename is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(out_filename, fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % skip_frames == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections

			class_IDs, scores, bounding_boxes = net(data.transforms.presets.ssd.transform_test(mx.nd.array(frame), 270)[0])

			# loop over the detections
			for i, (class_ID, score, bounding_box) in enumerate(zip(class_IDs[0], scores[0], bounding_boxes[0])):

				class_ID = int(class_ID[0].asnumpy()[0])
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = score[0].asnumpy()[0]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > confidence_threshold:
					# extract the index of the class label from the
					# detections list
					idx = int(class_ID)

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = bounding_box.asnumpy()
					(startX, startY, endX, endY) = box.astype("int")

					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				x = [c[0] for c in to.centroids]
				direction_y = centroid[1] - np.mean(y)
				direction_x = centroid[0] - np.mean(x)
				to.centroids.append(centroid)

				cur_x = np.mean(x)
				cur_y = np.mean(y)
				x_low, x_high, y_low, y_high = W // 3, 2 * W // 3, H // 4, 3 * H // 4
				
				cv2.line(frame, (x_low, y_low), (x_low, y_high), color=(0, 255, 0))
				cv2.line(frame, (x_high, y_low), (x_high, y_high), color=(0, 255, 0))
				cv2.line(frame, (x_high, y_low), (x_low, y_low), color=(0, 255, 0))
				cv2.line(frame, (x_high, y_high), (x_low, y_high), color=(0, 255, 0))

				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					delta_pixels = 10
					pred_x, pred_y = cur_x + delta_pixels * np.sign(direction_x), cur_y + delta_pixels * np.sign(direction_y)

					if ((cur_x < x_low or cur_x > x_high) and pred_x >= x_low and pred_x >= x_high) and ((cur_y < y_low or cur_y > y_high) and pred_y >= y_low and pred_y >= y_high):
						totalIn += 1
						to.counted = True
					elif cur_x >= x_low and cur_x <= x_high and cur_y >= y_low or cur_y <= y_high:
						totalOut += 1
						to.counted = True
					elif cur_x >= x_low and cur_x <= x_high and cur_y >= y_low and cur_y <= y_high and (pred_x < x_low or pred_x > x_high or pred_y < y_low or pred_y > y_high):
						totalOut += 1
						to.counted = True

			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			("In", totalIn),
			("Out", totalOut),
			("Status", status),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# check to see if we need to release the video writer pointer
	if writer is not None:
		writer.release()

	# if we are not using a video file, stop the camera video stream
	if not in_filename:
		vs.stop()

	# otherwise, release the video file pointer
	else:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()


if __name__ == '__main__':
	print("Zhopa")
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=True,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())
	make_prediction(
		in_filename=args.get("input"),
		out_filename=args.get("output"),
		caffe_prototxt_file=args.get("prototxt"),
		skip_frames=args.get("skip_frames"),
		confidence_threshold=args.get("confidence")
	)
