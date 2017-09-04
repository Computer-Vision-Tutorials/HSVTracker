#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;

int main(int, char**)
{
	// Open the default camera
    cv::VideoCapture cap(0);
    if(!cap.isOpened())
        return -1;

	// Mats to hold different parts of the image processing pipeline
	cv::Mat frame;
	cv::Mat hsv;
	cv::Mat mask;
	cv::Mat upperMask;
	cv::Mat lowerMask;
	cv::Mat maskedSaturation;

	// Split the 3 channels for the HSV image
	std::vector<cv::Mat> hsvChannels(3);

	// Holds contours for object detection
	std::vector<std::vector<cv::Point>> contours;

	// GUI window
    cv::namedWindow("Detected Object");

	bool isRunning = true;

	// Erosion kernel
	cv::Mat element = getStructuringElement(MORPH_RECT,
		cv::Size(7, 7),
		cv::Point(3, 3));

	// Drawing Color
	cv::Scalar color(255, 255, 0);

	// Array to hold the extremities of the object
	cv::Point2f vertices[4];

    while(isRunning)
    {
		// Get the next frame from the camera
        cap >> frame;

		// Blur
		cv::GaussianBlur(frame, frame, Size(7, 7), 1.5, 1.5);

		// Split frame into individual channels
		cv::cvtColor(frame, hsv, COLOR_BGR2HSV);
		cv::split(hsv, hsvChannels);

		// Filter for red image segments only
		cv::threshold(hsvChannels[0], upperMask, 160, 180, THRESH_BINARY);
		cv::threshold(hsvChannels[0], lowerMask, 0, 10, THRESH_BINARY);
		cv::bitwise_or(upperMask, lowerMask, mask);
		cv::bitwise_and(hsvChannels[1], cv::Scalar(255, 255, 255), maskedSaturation, mask = mask);

		// Remove noise
		cv::erode(maskedSaturation, maskedSaturation, element);

		// Threshold by saturation
		cv::threshold(maskedSaturation, maskedSaturation, 190, 255, THRESH_BINARY);

		// Find contours
		findContours(maskedSaturation, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Find largest contour by area
		int maxSize = 0;
		int maxIndex = -1;
		for (int i = 0; i < contours.size(); i++)
		{
			if (cv::contourArea(contours[i]) > maxSize) {
				maxIndex = i;
			}
		}

		// Draw largest contour
		if (maxIndex > -1) {
			cv::RotatedRect rotatedRect = cv::minAreaRect(contours[maxIndex]);
			rotatedRect.points(vertices);
			for (int i = 0; i < 4; i++)
				cv::line(frame, vertices[i], vertices[(i + 1) % 4], color, 3);
		}

		// Show image
        cv::imshow("Detected Object", frame);

        if(waitKey(1) >= 0) isRunning = false;
    }

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}