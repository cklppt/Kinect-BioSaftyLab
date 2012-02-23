// KinectHandTracking01.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>

//OpenNI c++
#include <XnCppWrapper.h>

#include <iostream>
#include <deque>
#include <stdio.h>

using namespace cv;
using namespace std;


enum Joints{ RHand, LHand, RElbow, LElbow};
enum Gestures{ Open, Closed};

int jointsX[4], jointsY[4], jointsDepth[4];
int gesture[2];
bool initialized = false;

extern "C" __declspec(dllexport) bool alreadyInit(){
	return initialized;
}

extern "C" __declspec(dllexport) int getX(int joint) {
	return jointsX[joint];
}

extern "C" __declspec(dllexport) int getY(int joint) {
	return jointsY[joint]; 
}

extern "C" __declspec(dllexport) int getDepth(int joint) {
	return jointsDepth[joint]; 
}

extern "C" __declspec(dllexport) int getGesture(int joint) {
	return gesture[joint];
}

void findConvexityDefects(vector<Point>& contour, vector<int>& hull, vector<Point>& convexDefects)
{

	if(hull.size() > 0 && contour.size() > 0){
		CvSeq* contourPoints;
		CvSeq* defects;
		CvMemStorage* storage;
		CvMemStorage* strDefects;
		CvMemStorage* contourStr;
		CvConvexityDefect *defectArray = 0;

		strDefects = cvCreateMemStorage();
		defects = cvCreateSeq( CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq),sizeof(CvPoint), strDefects );

		//We transform our vector<Point> into a CvSeq* object of CvPoint.
		contourStr = cvCreateMemStorage();
		contourPoints = cvCreateSeq(CV_SEQ_KIND_GENERIC|CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), contourStr);
		for(int i=0; i<(int)contour.size(); i++) {
			CvPoint cp = {contour[i].x,  contour[i].y};
			cvSeqPush(contourPoints, &cp);
		}

		//Now, we do the same thing with the hull index
		int count = (int)hull.size();
		//int hullK[count];
		int* hullK = (int*)malloc(count*sizeof(int));
		
		for(int i=0; i<count; i++)
		{
			hullK[i] = hull.at(i);
		}
		
		CvMat hullMat = cvMat(1, count, CV_32SC1, hullK);

		//We calculate convexity defects
		storage = cvCreateMemStorage(0);
		defects = cvConvexityDefects(contourPoints, &hullMat, storage);
		defectArray = (CvConvexityDefect*)malloc(sizeof(CvConvexityDefect)*defects->total);
		cvCvtSeqToArray(defects, defectArray, CV_WHOLE_SEQ);
		//printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y)

		//We store defects points in the convexDefects parameter.
		for(int i = 0; i<defects->total; i++){
			if (defectArray[i].depth > 8)
			{
				CvPoint ptf;
				ptf.x = defectArray[i].depth_point->x;
				ptf.y = defectArray[i].depth_point->y;
				convexDefects.push_back(ptf);
			}
		}

		//We release memory
		cvReleaseMemStorage(&contourStr);
		cvReleaseMemStorage(&strDefects);
		cvReleaseMemStorage(&storage);
	}
}

void handDetect(vector<Point>& contour, Mat& disparityMap, int whichHand)
{

	Mat distImage;
	distanceTransform(disparityMap, distImage,CV_DIST_L1, CV_DIST_MASK_PRECISE);

	Rect br = boundingRect(contour);
	CvBox2D smallRect = minAreaRect(contour);//minimum area bounding rectangle (possibly rotated)

	CvPoint2D32f smallRectPtsAr[4];
	cvBoxPoints(smallRect, smallRectPtsAr);
	vector<Point> smallRectPts;
	smallRectPts.push_back(smallRectPtsAr[0]);
	smallRectPts.push_back(smallRectPtsAr[1]);
	smallRectPts.push_back(smallRectPtsAr[2]);
	smallRectPts.push_back(smallRectPtsAr[3]);

	Mat subMat = distImage(boundingRect(contour));
	float biggestDist = -1;
	double mi, mj;
	int nMaxPts = 0;

	for (int i = 0; i < subMat.rows; i++) //i = y pixel coordinate (from upper left of hand contour bounding box)
	{
		for (int j = 0; j < subMat.cols; j++) // j = x pixel coord. (from upper left)
		{
			float val = subMat.at<float>(i,j);// - (i*.05);//i term "punishes" points low on the image

			//Penalize points "down" the arm (towards the elbow)
			//  Actually, should probably do some convexity analysis to determine
			// which is the hand end of the contour.
			// [!!!] Use Kuan's code to correctly differentiate wrist/elbow end of contour
			if (i > 80) //maybe midpoint of hand and elbow. but need to consider the case when forearm is horizontal
			{
				val -= 100;
			}

			if (val >= biggestDist)
			{
				//double dist = pointPolygonTest(contours0[idxOfBiggestArea], Point(br.x + j, br.y + i), CV_DIST_L1);
				double rectDist = pointPolygonTest(smallRectPts, Point(br.x + j, br.y + i), CV_DIST_L1);//nearest distance to minimum area bounding rect
				if (rectDist > 0 /*inside bounding rect*/ && pointPolygonTest(contour, Point(br.x + j, br.y + i), CV_DIST_L1) > 0 /*inside contour*/)
				{

					//if (val >= biggestDist)  //already an if (val >= biggestDist) above
					//{
					if (val > biggestDist)
					{
						biggestDist = val; 
						mi = i;
						mj = j;
						nMaxPts = 1;
					}
					else //val == biggestDist
					{
						mi += i;
						mj += j;
						nMaxPts++;
					}
					//}
				}
			}
		}
	}

	//average coordinate of biggest distance points <= palm center
	mi /= nMaxPts;
	mj /= nMaxPts;

	Point palmCenter(br.x + mj, br.y + mi);
	if(whichHand == RHand)
	{
		jointsX[RHand] = palmCenter.x;
		jointsY[RHand] = palmCenter.y;
	}
	else if(whichHand == LHand)
	{
		jointsX[LHand] = palmCenter.x;
		jointsY[LHand] = palmCenter.y;
	}

	circle(disparityMap, palmCenter, 15, cvScalar(0,0,0));

	//-------------- Do some convexity analysis

	vector<Point> biggestContour = contour;
	vector<vector<Point> > biggestHull;
	vector<Point> hull(biggestContour);
	vector<int> hullIndices;

	convexHull(biggestContour, hull);
	convexHull(biggestContour, hullIndices, false, false);

	biggestHull.push_back(hull);
	drawContours( disparityMap, biggestHull, 0, Scalar(0), 2);
	vector<Point> convexityDefects;

	//[!!!] Currrently getting thresholded! fix function to return vector of defect objects!
	findConvexityDefects(biggestContour, hullIndices, convexityDefects);

	int nDefectsNearPalm = 0;
	for (int i = 0; i < convexityDefects.size(); i++)
	{
		circle(disparityMap, convexityDefects[i], 5, cvScalar(0,0,0));
		double distFromPalm = norm(palmCenter - convexityDefects[i]);
		if (distFromPalm < 100)//base on the length of the forearm?
		{
			nDefectsNearPalm++;
		}
	}

	bool closedFist = nDefectsNearPalm < 4;//4 works fine on me...
	if (closedFist)
	{
		gesture[whichHand] = Closed;
		circle(disparityMap, palmCenter, 12, cvScalar(0,0,0), -1);
	}
	else
		gesture[whichHand] = Open;

	//Point maxPts[100];
	//minMaxLoc(disparityMap(boundingRect(contours0[idxOfBiggestArea])), NULL, NULL, NULL);

}



float getMaxDisparity( VideoCapture& capture )
{
	const int minDistance = 400; // mm
	float b = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
	float F = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
	return b * F / minDistance;
}


//OpenNI Callback functions
//callback function of user generator: new user
void XN_CALLBACK_TYPE NewUser( xn::UserGenerator& generator, XnUserID user, void* pCookie )
{
	cout << "New user identified: " << user << endl;
	generator.GetSkeletonCap().RequestCalibration( user, true );
}
// callback function of skeleton: calibration end 
void XN_CALLBACK_TYPE CalibrationEnd( xn::SkeletonCapability& skeleton, XnUserID user, XnCalibrationStatus eStatus, void* pCookie )
{
	cout << "Calibration complete for user " <<  user << ", ";
	if( eStatus == XN_CALIBRATION_STATUS_OK )
	{
		cout << "Success" << endl;
		skeleton.StartTracking( user );
	}
	else
	{
		cout << "Failure" << endl;
		skeleton.RequestCalibration( user, true );
	}
}


//OpenNI
xn::Context mContext;
xn::DepthMetaData m_DepthMD;
xn::UserGenerator mUserGenerator;
xn::SkeletonCapability mSC(mUserGenerator);
xn::DepthGenerator mDepthGenerator;
VideoCapture capture;

//Joints data
//enum Joints{ RHand, RElbow, LHand, LElbow};
//Raw joints data include position and orientation
XnSkeletonJointTransformation JointsTransformation[4];
//Joints in Realworld coordinates and Screen coordinates
XnPoint3D  JointsReal[4];
XnPoint3D  JointsScreen[4];
//Length form elbow to hand in real world(mm)
//int RLength = 0, LLength = 0;

extern "C" __declspec(dllexport) int initKinect(){

	cout << "Kinect opening ..." << endl;
	capture = VideoCapture(CV_CAP_OPENNI);
	cout << "done." << endl;

	if( !capture.isOpened() )
	{
		cout << "Can not open a capture object." << endl;
		return -1;
	}

	capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_VGA_30HZ ); // default

	// Print some avalible Kinect settings.
	cout << "\nDepth generator output mode:" << endl <<
		"FRAME_WIDTH    " << capture.get( CV_CAP_PROP_FRAME_WIDTH ) << endl <<
		"FRAME_HEIGHT   " << capture.get( CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
		"FRAME_MAX_DEPTH    " << capture.get( CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
		"FPS    " << capture.get( CV_CAP_PROP_FPS ) << endl;

	cout << "\nImage generator output mode:" << endl <<
		"FRAME_WIDTH    " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_WIDTH ) << endl <<
		"FRAME_HEIGHT   " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FRAME_HEIGHT ) << endl <<
		"FPS    " << capture.get( CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_FPS ) << endl;

	//store a brief timeline of the depths to get a more robust detection of a 'click'
	deque<int> depths;

	//OpenNI
	//initial context
	mContext.Init();

	//create user generator
	mUserGenerator.Create( mContext );

	//create depth generator
	XnMapOutputMode mapMode;
	mapMode.nXRes = 640;  
	mapMode.nYRes = 480; 
	mapMode.nFPS = 30; 
	mDepthGenerator.Create( mContext ); 
	mDepthGenerator.SetMapOutputMode( mapMode );  

	//Register callback functions of user generator
	XnCallbackHandle hUserCB;
	mUserGenerator.RegisterUserCallbacks( NewUser, NULL, NULL, hUserCB );

	//Register callback functions of skeleton capability
	mSC = mUserGenerator.GetSkeletonCap();
	mSC.SetSkeletonProfile( XN_SKEL_PROFILE_UPPER);//XN_SKEL_PROFILE_UPPER  XN_SKEL_PROFILE_HEAD_HANDS
	XnCallbackHandle hCalibCB;
	mSC.RegisterToCalibrationComplete( CalibrationEnd, &mUserGenerator, hCalibCB );

	initialized = true;

	//start generate data
	mContext.StartGeneratingAll();
}

extern "C" __declspec(dllexport) int closeKinect(){
	mContext.StopGeneratingAll();
	capture.release();
	mDepthGenerator.Release();
	mUserGenerator.Release();
	mContext.Release();
	initialized = false;
	return 0;
}

/*
* To work with Kinect the user must install OpenNI library and PrimeSensorModule for OpenNI and
* configure OpenCV with WITH_OPENNI flag is ON (using CMake).
*/
extern "C" __declspec(dllexport) int updateKinect()
{

	Mat depthMap;
	Mat validDepthMap;
	Mat disparityMap;
	Mat bgrImage;
	Mat grayImage;

	int thresholdHand = 65;
	int thickOfThreshold = 3;//the thick of threshold based on hand joints
	int handRadius = 110;//the radius of the circle form hand joints
	XnUserID userID = 0;

	Point joints[4];

	//OpenNI
	Mat m_depthmap(480, 640, CV_8UC3 );
	Mat m_depth16u(480, 640, CV_16UC1 );
	//Update date
	mContext.WaitAndUpdateAll();
	//cout << "updated" << endl;

	// 7. get user information
	XnUInt16 nUsers = mUserGenerator.GetNumberOfUsers();
	if( nUsers > 0 )
	{
		//cout << "get user" << endl;

		// 8. get users
		XnUserID* aUserID = new XnUserID[nUsers];
		mUserGenerator.GetUsers( aUserID, nUsers );

		// 9. check each user //Now only use one user
		for( int i = 0; i < nUsers; ++i )
		{
			// 10. if is tracking skeleton
			if( mSC.IsTracking( aUserID[i] ) )
			{
				// 11. get skeleton joint data
				mSC.GetSkeletonJoint( aUserID[i], XN_SKEL_RIGHT_HAND, JointsTransformation[LHand] );
				mSC.GetSkeletonJoint( aUserID[i], XN_SKEL_RIGHT_ELBOW, JointsTransformation[LElbow] );
				mSC.GetSkeletonJoint( aUserID[i], XN_SKEL_LEFT_HAND, JointsTransformation[RHand] );
				mSC.GetSkeletonJoint( aUserID[i], XN_SKEL_LEFT_ELBOW, JointsTransformation[RElbow] );

				for(int i=0; i<4; i++){
					JointsReal[i] = xnCreatePoint3D(JointsTransformation[i].position.position.X, JointsTransformation[i].position.position.Y, JointsTransformation[i].position.position.Z);
				}

				//Calc distance between hand & elbow
				/*
				RLength = sqrt(pow(JointsReal[RHand].X-JointsReal[RElbow].X, 2) + 
				pow(JointsReal[RHand].Y-JointsReal[RElbow].Y, 2) + 
				pow(JointsReal[RHand].Z-JointsReal[RElbow].Z, 2));

				LLength = sqrt(pow(JointsReal[LHand].X-JointsReal[LElbow].X, 2) + 
				pow(JointsReal[LHand].Y-JointsReal[LElbow].Y, 2) + 
				pow(JointsReal[LHand].Z-JointsReal[LElbow].Z, 2));
				*/

				//Convert from realworld coordinates to Projective(Screen) coordinates
				mDepthGenerator.ConvertRealWorldToProjective(4, JointsReal, JointsScreen);

				//Convert from depth to disparity:  disparity = baseline * Focallength / z(depth);
				float b = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
				float f = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
				float mult = b /*mm*/ * f /*pixels*/;

				for(int i=0; i<4; i++){
					JointsScreen[i].Z = (mult / JointsScreen[i].Z);
					joints[i] = Point(JointsScreen[i].X, JointsScreen[i].Y);
				}

				if(JointsScreen[RHand].X < 0 || JointsScreen[RHand].Y < 0)
					JointsScreen[RHand].Z = 0xFF;//if out of window, don't use the depth
				if(JointsScreen[LHand].X < 0 || JointsScreen[LHand].Y < 0)
					JointsScreen[LHand].Z = 0xFF;//if out of window, don't use the depth
				//0(far)-65(near)   20(3m) - 85(1m)
				//take the farest hand as threshold
				thresholdHand = (JointsScreen[RHand].Z < JointsScreen[LHand].Z ? JointsScreen[RHand].Z : JointsScreen[LHand].Z) -  thickOfThreshold;

				userID = aUserID[i];
			}
		}
		delete [] aUserID;
	}

	if( !capture.grab() )
	{
		cout << "Can not grab images." << endl;
		return -1;
	}
	else
	{
		if(capture.retrieve( disparityMap, CV_CAP_OPENNI_DISPARITY_MAP ) )
		{
			//cout << "start capturing" << endl;
			//Show the OpenNI depth map
			// 9a. get the depth map  
			mDepthGenerator.GetMetaData(m_DepthMD);
			memcpy(m_depth16u.data, m_DepthMD.Data(),640*480*2);

			//16U Convert to 8U
			m_depth16u.convertTo(m_depthmap, CV_8U, 255.0/2550.0);
			cvtColor(m_depthmap,m_depthmap,CV_GRAY2RGB);

			bool isTracking = mSC.IsTracking(userID);

			//draw lines of the arms
			if(isTracking){
				cvLine(&(m_depthmap.operator IplImage()), cvPoint(JointsScreen[RElbow].X, JointsScreen[RElbow].Y), cvPoint(JointsScreen[RHand].X, JointsScreen[RHand].Y), CV_RGB(0,255,0), 3, 8, 0);
				cvLine(&(m_depthmap.operator IplImage()), cvPoint(JointsScreen[LElbow].X, JointsScreen[LElbow].Y), cvPoint(JointsScreen[LHand].X, JointsScreen[LHand].Y), CV_RGB(0,255,0), 3, 8, 0);
			}
			imshow( "depthmap", m_depthmap );

			Mat colorDisparityMap;
			Mat filterScratch;

			/*//Debug
			if(isTracking){
			char temp[10];
			CvFont Font;
			cvInitFont( &Font,CV_FONT_HERSHEY_SIMPLEX,0.5,0.5,0.0,1, CV_AA );
			sprintf(temp,"(%d,%d)", (int)JointsScreen[RHand].X, (int)JointsScreen[RHand].Y);
			cvPutText(&(m_depthmap.operator IplImage()), temp, cvPoint(20, 20), &Font, CV_RGB(0,255,0));
			if(JointsScreen[RHand].X > 0 && JointsScreen[RHand].X < 640 && JointsScreen[RHand].Y > 0 && JointsScreen[RHand].Y < 480)
			sprintf(temp,"%d",(int)disparityMap.at<unsigned short>((int)JointsScreen[RHand].X, (int)JointsScreen[RHand].Y));
			cvPutText(&(m_depthmap.operator IplImage()), temp, cvPoint(20, 50), &Font, CV_RGB(255,0,0));
			circle(disparityMap, Point(JointsScreen[RHand].X, JointsScreen[RHand].Y), 20, cvScalar(255), 4);
			}
			imshow( "disparityMap", disparityMap );
			*/

			/*
			if(isTracking){
			for (int y = 0; y < disparityMap.rows; y++)
			{
			for (int x = 0; x < disparityMap.cols; x++)
			{
			if( norm( Point(x,y) - Point(joints[RHand].x, joints[RHand].y) ) > 150 )
			disparityMap.at<unsigned short>(x, y) = 0;
			}
			}
			}
			*/

			blur(disparityMap, filterScratch, Size(5, 5));
			dilate(filterScratch, disparityMap, Mat(),Point(-1,-1),2);
			threshold(disparityMap, disparityMap, thresholdHand, 255, THRESH_TOZERO);//cut 0(far)-65(near)   20(3m) - 85(1m)

			//Draw circles on the hand joints, cut everything outside the circle
			if(isTracking){
				for (int y = 0; y < disparityMap.rows; y++) {
					for (int x = 0; x < disparityMap.cols; x++) {
						if(disparityMap.at<unsigned char>(y, x) != 0) {
							if( norm(Point(x,y) - Point(joints[RHand].x, joints[RHand].y)) > handRadius &&
								norm(Point(x,y) - Point(joints[LHand].x, joints[LHand].y)) > handRadius )
								disparityMap.at<unsigned char>(y, x) = 0;
						}
					}
				}
			}

			vector<vector<Point> > contours0;
			vector<vector<Point> > hull;
			vector<Vec4i> hierarchy;

			//unnecessary code? what does it do?
			/*
			colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp ? getMaxDisparity(capture) : -1 );

			Mat validColorDisparityMap;

			colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
			*/

			//get the contours

			threshold(disparityMap, filterScratch, thresholdHand, 255, THRESH_BINARY);
			findContours( filterScratch, contours0, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			//Get average position, weight by y-value:
			/*
			double sum=0;

			int avX = 0;
			int avY = 0;

			bool foundHighest = false;

			//idea 1:
			//Make histogram of total nPixels > thresh higher than a particular height
			int heightHist[600];
			int newDepth = -1;

			for(int i = 0; i < validColorDisparityMap.rows && !foundHighest; i++)
			{
			//const double* row = validColorDisparityMap.ptr<double>(i);
			double multiplier = (validColorDisparityMap.rows - i + 0.0) / validColorDisparityMap.rows;
			multiplier = std::pow(multiplier, 1);

			if (i > 0)
			{
			heightHist[i] = heightHist[i - 1];
			}

			for(int j = 0; j < validColorDisparityMap.cols; j++)
			{
			//uchar val = row[j];
			double val = disparityMap.at<uchar>(i,j) / 255.0;
			//val = std::pow(val, 2);

			if (val > 0)
			{
			newDepth = disparityMap.at<uchar>(i,j);

			//update the depth history
			depths.push_back(newDepth);
			if ( depths.size() > CLICK_WINDOW )
			{
			depths.pop_front();
			}

			heightHist[i] += 1; //should I weight?
			foundHighest = true;
			sum += multiplier * val;
			avX += j * multiplier * val;
			avY += i * multiplier * val;
			}
			}
			}

			avX /= sum;
			avY /= sum;

			double depthSpeed = 0;
			for (int i = 0; i + 1 < depths.size() && depths.size() == CLICK_WINDOW; i++)
			{
			depthSpeed += (depths[i+1] - depths[i])/2.0/depths.size();
			}

			bool clickDetected = depthSpeed > 0.05;//newDepth - prevDepth >= 2 && prevDepth != -1;
			if ( clickDetected )
			{
			//circle(validColorDisparityMap, Point(avX, avY), 40, cvScalar(0,0,255,255));
			}
			else
			{
			//circle(validColorDisparityMap, Point(avX, avY), 50, cvScalar(255,255,255,255));
			}
			*/

			//  Horizontally flip (mirror) image

			Mat flipped;
			Mat distImage;

			//Find the contours which hand joints are within
			int idxOfRHand = -1;
			int idxOfLHand = -1;

			threshold(disparityMap, disparityMap, -1, 255, THRESH_BINARY);//clean all

			for (int i = 0; i < contours0.size(); i++)
			{
				if(pointPolygonTest(contours0[i],Point(joints[RHand].x, joints[RHand].y), true) > -5)//hand joint "near"(not just inside) the contour polygon
					idxOfRHand = i;

				if(pointPolygonTest(contours0[i],Point(joints[LHand].x, joints[LHand].y), true) > -5)//hand joint "near"(not just inside) the contour polygon
					idxOfLHand = i;

				drawContours( disparityMap, contours0, i, Scalar(0), 1);
			}

			if (idxOfRHand >= 0)
				handDetect(contours0[idxOfRHand], disparityMap, RHand);
			if (idxOfLHand >= 0)
				handDetect(contours0[idxOfLHand], disparityMap, LHand);

			//draw lines of the forearms

			if(isTracking){
				cvLine(&(disparityMap.operator IplImage()), cvPoint(JointsScreen[RElbow].X, JointsScreen[RElbow].Y), cvPoint(JointsScreen[RHand].X, JointsScreen[RHand].Y), CV_RGB(0,255,0), 3, 8, 0);
				cvLine(&(disparityMap.operator IplImage()), cvPoint(JointsScreen[LElbow].X, JointsScreen[LElbow].Y), cvPoint(JointsScreen[LHand].X, JointsScreen[LHand].Y), CV_RGB(0,255,0), 3, 8, 0);
				circle(disparityMap, Point(JointsScreen[RHand].X, JointsScreen[RHand].Y), 5, cvScalar(0));
			}

			flip(disparityMap, flipped, 1); //flip horizontal

			//debug: show the depth value of right hand joint
			/*
			if(isTracking){
			char temp[10];
			CvFont Font;
			cvInitFont( &Font,CV_FONT_HERSHEY_SIMPLEX,0.5,0.5,0.0,1, CV_AA );
			sprintf(temp,"Z(R:%d",(int)JointsScreen[RHand].Z);
			cvPutText(&(flipped.operator IplImage()), temp, cvPoint(20, 20), &Font, CV_RGB(255,0,0));
			sprintf(temp,"TH(R:%d", thresholdHand);
			cvPutText(&(flipped.operator IplImage()), temp, cvPoint(20, 100), &Font, CV_RGB(255,0,0));
			}
			*/

			imshow( "colorized disparity map", flipped );

		}

		if( waitKey( 30 ) >= 0 )
			return 0;

	}

	return 0;
}


int getRHandX()
{
	return JointsScreen[RHand].X;
}

int getRHandY()
{
	return JointsScreen[RHand].Y;
}

int getLHandX()
{
	return JointsScreen[RHand].X;
}

int getLHandY()
{
	return JointsScreen[RHand].Y;
}