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

#define CLICK_WINDOW 25


void help()
{
	cout << "\nThis program demonstrates usage of Kinect sensor.\n"
		"The user gets some of the supported output images.\n"
		"\nAll supported output map types:\n"
		"1.) Data given from depth generator\n"
		"   OPENNI_DEPTH_MAP            - depth values in mm (CV_16UC1)\n"
		"   OPENNI_POINT_CLOUD_MAP      - XYZ in meters (CV_32FC3)\n"
		"   OPENNI_DISPARITY_MAP        - disparity in pixels (CV_8UC1)\n"
		"   OPENNI_DISPARITY_MAP_32F    - disparity in pixels (CV_32FC1)\n"
		"   OPENNI_VALID_DEPTH_MASK     - mask of valid pixels (not ocluded, not shaded etc.) (CV_8UC1)\n"
		"2.) Data given from RGB image generator\n"
		"   OPENNI_BGR_IMAGE            - color image (CV_8UC3)\n"
		"   OPENNI_GRAY_IMAGE           - gray image (CV_8UC1)\n"
		<< endl;
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
		//printf("DefectArray %i %i\n",defectArray->end->x, defectArray->end->y);

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


void colorizeDisparity( const Mat& gray, Mat& rgb, double maxDisp=-1.f, float S=1.f, float V=1.f )
{
	CV_Assert( !gray.empty() );
	CV_Assert( gray.type() == CV_8UC1 );

	if( maxDisp <= 0 )
	{
		maxDisp = 0;
		minMaxLoc( gray, 0, &maxDisp );
	}

	rgb.create( gray.size(), CV_8UC3 );
	rgb = Scalar::all(0);
	if( maxDisp < 1 )
		return;

	for( int y = 0; y < gray.rows; y++ )
	{
		for( int x = 0; x < gray.cols; x++ )
		{
			uchar d = gray.at<uchar>(y,x);

			if ( d > 0 )
			{
				double dd = d - 65.0;
				dd *= (400/(255 - 65));
				d = (uchar) dd;
			}

			unsigned int H = ((uchar)maxDisp - d) * 240 / (uchar)maxDisp;

			unsigned int hi = (H/60) % 6;
			float f = H/60.f - H/60;
			float p = V * (1 - S);
			float q = V * (1 - f * S);
			float t = V * (1 - (1 - f) * S);

			Point3f res;

			if( hi == 0 ) //R = V,	G = t,	B = p
				res = Point3f( p, t, V );
			if( hi == 1 ) // R = q,	G = V,	B = p
				res = Point3f( p, V, q );
			if( hi == 2 ) // R = p,	G = V,	B = t
				res = Point3f( t, V, p );
			if( hi == 3 ) // R = p,	G = q,	B = V
				res = Point3f( V, q, p );
			if( hi == 4 ) // R = t,	G = p,	B = V
				res = Point3f( V, p, t );
			if( hi == 5 ) // R = V,	G = p,	B = q
				res = Point3f( q, p, V );

			//uchar b = (uchar)(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
			//uchar g = (uchar)(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
			//uchar r = (uchar)(std::max(0.f, std::min (res.z, 1.f)) * 255.f);

			//rgb.at<Point3_<uchar> >(y,x) = Point3_<uchar>(b, g, r);     
		}
	}
}

float getMaxDisparity( VideoCapture& capture )
{
	const int minDistance = 400; // mm
	float b = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE ); // mm
	float F = (float)capture.get( CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH ); // pixels
	return b * F / minDistance;
}

void printCommandLineParams()
{
	cout << "-cd       Colorized disparity? (0 or 1; 1 by default) Ignored if disparity map is not selected to show." << endl;
	cout << "-fmd      Fixed max disparity? (0 or 1; 0 by default) Ignored if disparity map is not colorized (-cd 0)." << endl;
	cout << "-sxga     SXGA resolution of image? (0 or 1; 0 by default) Ignored if rgb image or gray image are not selected to show." << endl;
	cout << "          If -sxga is 0 then vga resolution will be set by default." << endl;
	cout << "-m        Mask to set which output images are need. It is a string of size 5. Each element of this is '0' or '1' and" << endl;
	cout << "          determine: is depth map, disparity map, valid pixels mask, rgb image, gray image need or not (correspondently)?" << endl ;
	cout << "          By default -m 01010 i.e. disparity map and rgb image will be shown." << endl ;
}

void parseCommandLine( int argc, char* argv[], bool& isColorizeDisp, bool& isFixedMaxDisp, bool& isSetSXGA, bool retrievedImageFlags[] )
{
	// set defaut values
	isColorizeDisp = true;
	isFixedMaxDisp = false;
	isSetSXGA = false;

	retrievedImageFlags[0] = false;
	retrievedImageFlags[1] = true;
	retrievedImageFlags[2] = false;
	retrievedImageFlags[3] = true;
	retrievedImageFlags[4] = false;

	if( argc == 1 )
	{
		help();
	}
	else
	{
		for( int i = 1; i < argc; i++ )
		{
			if( !strcmp( argv[i], "--help" ) || !strcmp( argv[i], "-h" ) )
			{
				printCommandLineParams();
				exit(0);
			}
			else if( !strcmp( argv[i], "-cd" ) )
			{
				isColorizeDisp = atoi(argv[++i]) == 0 ? false : true;
			}
			else if( !strcmp( argv[i], "-fmd" ) )
			{
				isFixedMaxDisp = atoi(argv[++i]) == 0 ? false : true;
			}
			else if( !strcmp( argv[i], "-sxga" ) )
			{
				isSetSXGA = atoi(argv[++i]) == 0 ? false : true;
			}
			else if( !strcmp( argv[i], "-m" ) )
			{
				string mask( argv[++i] );
				if( mask.size() != 5)
					CV_Error( CV_StsBadArg, "Incorrect length of -m argument string" );
				int val = atoi(mask.c_str());

				int l = 100000, r = 10000, sum = 0;
				for( int i = 0; i < 5; i++ )
				{
					retrievedImageFlags[i] = ((val % l) / r ) == 0 ? false : true;
					l /= 10; r /= 10;
					if( retrievedImageFlags[i] ) sum++;
				}

				if( sum == 0 )
				{
					cout << "No one output image is selected." << endl;
					exit(0);
				}
			}
			else
			{
				cout << "Unsupported command line argument: " << argv[i] << "." << endl;
				exit(-1);
			}
		}
	}
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


/*
* To work with Kinect the user must install OpenNI library and PrimeSensorModule for OpenNI and
* configure OpenCV with WITH_OPENNI flag is ON (using CMake).
*/
int main( int argc, char* argv[] )
{
	bool isColorizeDisp, isFixedMaxDisp, isSetSXGA;
	bool retrievedImageFlags[CLICK_WINDOW];
	parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, isSetSXGA, retrievedImageFlags );

	cout << "Kinect opening ..." << endl;
	VideoCapture capture( CV_CAP_OPENNI );
	cout << "done." << endl;

	if( !capture.isOpened() )
	{
		cout << "Can not open a capture object." << endl;
		return -1;
	}

	if( isSetSXGA )
		capture.set( CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CV_CAP_OPENNI_SXGA_15HZ );
	else
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

	//Joints data
	enum Joints{ RHand, RElbow, LHand, LElbow};
	//Raw joints data include position and orientation
	XnSkeletonJointTransformation JointsTransformation[4];
	//Joints in Realworld coordinates and Screen coordinates
	XnPoint3D  JointsReal[4];
	XnPoint3D  JointsScreen[4];
	//Length form elbow to hand in real world(mm)
	int RLength = 0, LLength = 0;

	//OpenNI
	//initial context
	xn::Context mContext;
	mContext.Init();

	//create user generator
	xn::UserGenerator mUserGenerator;
	mUserGenerator.Create( mContext );

	//create depth generator
	xn::DepthMetaData m_DepthMD;
	XnMapOutputMode mapMode; 
	mapMode.nXRes = 640;  
	mapMode.nYRes = 480; 
	mapMode.nFPS = 30; 
	xn::DepthGenerator mDepthGenerator;  
	mDepthGenerator.Create( mContext ); 
	mDepthGenerator.SetMapOutputMode( mapMode );  

	//Register callback functions of user generator
	XnCallbackHandle hUserCB;
	mUserGenerator.RegisterUserCallbacks( NewUser, NULL, NULL, hUserCB );

	//Register callback functions of skeleton capability
	xn::SkeletonCapability mSC = mUserGenerator.GetSkeletonCap();
	mSC.SetSkeletonProfile( XN_SKEL_PROFILE_UPPER);//XN_SKEL_PROFILE_HEAD_HANDS
	XnCallbackHandle hCalibCB;
	mSC.RegisterToCalibrationComplete( CalibrationEnd, &mUserGenerator, hCalibCB );

	//start generate data
	mContext.StartGeneratingAll();

	//OpenNI calc Disparity
	//Convert from depth to disparity:  disparity = baseline * Focallength / z(depth);
	// Distance between IR projector and IR camera (in meters)
	XnDouble baseline;
	// Focal length for the IR camera in VGA resolution (in pixels)
	XnUInt64 depthFocalLength_VGA;

	//baseline
	mDepthGenerator.GetRealProperty( "LDDIS", baseline );
	// baseline from cm -> mm
	baseline *= 10;

	//focal length
	XnUInt64 zeroPlanDistance;
	mDepthGenerator.GetIntProperty( "ZPD", zeroPlanDistance );
	XnDouble pixelSize = 0;
	mDepthGenerator.GetRealProperty( "ZPPS", pixelSize );
	// focal length from mm -> pixels (valid for 640x480)
	depthFocalLength_VGA = (XnUInt64)((double)zeroPlanDistance / (double)pixelSize);

	float mult = baseline /*mm*/ * depthFocalLength_VGA /*pixels*/;

	for(;;)
	{
		Mat depthMap;
		Mat validDepthMap;
		Mat disparityMap;
		Mat bgrImage;
		Mat grayImage;

		int thresholdRHand = 65;

		//OpenNI
		Mat m_depthmap( 480,640,CV_8UC3);
		Mat  m_depth16u( 480,640,CV_16UC1);
		//Update date
		mContext.WaitAndUpdateAll();

		// 7. get user information
		XnUInt16 nUsers = mUserGenerator.GetNumberOfUsers();
		if( nUsers > 0 )
		{
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
					
					RLength = sqrt(pow(JointsReal[RHand].X-JointsReal[RElbow].X, 2) + 
						pow(JointsReal[RHand].Y-JointsReal[RElbow].Y, 2) + 
						pow(JointsReal[RHand].Z-JointsReal[RElbow].Z, 2));

					LLength = sqrt(pow(JointsReal[LHand].X-JointsReal[LElbow].X, 2) + 
						pow(JointsReal[LHand].Y-JointsReal[LElbow].Y, 2) + 
						pow(JointsReal[LHand].Z-JointsReal[LElbow].Z, 2));
					

					//Convert from realworld coordinates to Projective(Screen) coordinates
					mDepthGenerator.ConvertRealWorldToProjective(4, JointsReal, JointsScreen);

					//Convert from depth to disparity:  disparity = baseline * Focallength / z(depth);
					for(int i=0; i<4; i++){
						JointsScreen[i].Z = (mult / JointsScreen[i].Z);
					}
					//cout << "Is trackingLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL" << endl;
					thresholdRHand = 255 - (JointsScreen[RHand].Z - 35);
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
			if( retrievedImageFlags[0] && capture.retrieve( depthMap, CV_CAP_OPENNI_DEPTH_MAP ) )
			{
				const float scaleFactor = 0.05f;
				Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
				imshow( "depth map", show );
			}

			if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CV_CAP_OPENNI_DISPARITY_MAP ) )
			{

				//Show the OpenNI depth map
				// 9a. get the depth map  
				mDepthGenerator.GetMetaData(m_DepthMD);
				memcpy(m_depth16u.data, m_DepthMD.Data(),640*480*2);

				//16U Convert to 8U
				m_depth16u.convertTo(m_depthmap, CV_8U, 255.0/2550.0);
				cvtColor(m_depthmap,m_depthmap,CV_GRAY2RGB);

				//draw lines of the arms
				if(mUserGenerator.GetNumberOfUsers() > 0 && RLength && LLength){
					cvLine(&(m_depthmap.operator IplImage()), cvPoint(JointsScreen[1].X, JointsScreen[1].Y), cvPoint(JointsScreen[0].X, JointsScreen[0].Y), CV_RGB(0,255,0), 3, 8, 0);
					cvLine(&(m_depthmap.operator IplImage()), cvPoint(JointsScreen[3].X, JointsScreen[3].Y), cvPoint(JointsScreen[2].X, JointsScreen[2].Y), CV_RGB(0,255,0), 3, 8, 0);
				}
				
				imshow( "depthmap", m_depthmap );
				

				Mat colorDisparityMap;
				Mat filterScratch;

				blur(disparityMap, filterScratch, Size(5, 5));
				dilate(filterScratch, disparityMap, Mat(),Point(-1,-1),2);
				threshold(disparityMap, disparityMap, 65, 255, THRESH_TOZERO); 

				vector<vector<Point> > contours0;
				vector<vector<Point> > hull;
				vector<Vec4i> hierarchy;

				colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp ? getMaxDisparity(capture) : -1 );

				Mat validColorDisparityMap;

				colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );

				//get the contours
				threshold(disparityMap, filterScratch, 65, 255, THRESH_BINARY);
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


				//find biggest contour

				int biggestArea = -1;
				int idxOfBiggestArea = -1;

				threshold(disparityMap, disparityMap, -1, 255, THRESH_BINARY);
				for (int i = 0; i < contours0.size(); i++)
				{
					if(boundingRect(contours0[i]).area() >= 4000)
					{
						if (boundingRect(contours0[i]).area() > biggestArea)
						{
							biggestArea = boundingRect(contours0[i]).area();
							idxOfBiggestArea = i;
						}

						//drawContours( validColorDisparityMap, contours0, i, Scalar(255,255,255), 2);
						drawContours( disparityMap, contours0, i, Scalar(0), 1);
					}
				}




				distanceTransform(disparityMap, distImage,CV_DIST_L1, CV_DIST_MASK_PRECISE);
				if (idxOfBiggestArea >= 0)
				{
					Rect br = boundingRect(contours0[idxOfBiggestArea]);
					CvBox2D smallRect = minAreaRect(contours0[idxOfBiggestArea]);

					CvPoint2D32f smallRectPtsAr[4];
					cvBoxPoints(smallRect, smallRectPtsAr);
					vector<Point> smallRectPts;
					smallRectPts.push_back(smallRectPtsAr[0]);
					smallRectPts.push_back(smallRectPtsAr[1]);
					smallRectPts.push_back(smallRectPtsAr[2]);
					smallRectPts.push_back(smallRectPtsAr[3]);

					Mat subMat = distImage(boundingRect(contours0[idxOfBiggestArea]));
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
							if (i > 100)								
							{
								val -= 100;
							}
							
							if (val >= biggestDist)
							{
								//double dist = pointPolygonTest(contours0[idxOfBiggestArea], Point(br.x + j, br.y + i), CV_DIST_L1);
								double rectDist = pointPolygonTest(smallRectPts, Point(br.x + j, br.y + i), CV_DIST_L1);
								if (rectDist > 0 && pointPolygonTest(contours0[idxOfBiggestArea], Point(br.x + j, br.y + i), CV_DIST_L1) > 0
									)
								{

									if (val >= biggestDist)
									{
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
									}
								}
							}
						}
					}

					mi /= nMaxPts;
					mj /= nMaxPts;

					Point palmCenter(br.x + mj, br.y + mi);
					circle(disparityMap, palmCenter, 15, cvScalar(0,0,0));

					//-------------- Do some convexity analysis

					vector<Point> biggestContour = contours0[idxOfBiggestArea];
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
						if (distFromPalm < 150)
						{
							nDefectsNearPalm++;
						}
					}

					bool closedFist = nDefectsNearPalm < 5;
					if (closedFist)
					{
						circle(disparityMap, palmCenter, 12, cvScalar(0,0,0), -1);
					}

					//Point maxPts[100];
					//minMaxLoc(disparityMap(boundingRect(contours0[idxOfBiggestArea])), NULL, NULL, NULL);
				}

				//draw lines of the arms
				if(mUserGenerator.GetNumberOfUsers() > 0 && RLength && LLength){
					cvLine(&(disparityMap.operator IplImage()), cvPoint(JointsScreen[RElbow].X, JointsScreen[RElbow].Y), cvPoint(JointsScreen[RHand].X, JointsScreen[RHand].Y), CV_RGB(0,255,0), 3, 8, 0);
					cvLine(&(disparityMap.operator IplImage()), cvPoint(JointsScreen[LElbow].X, JointsScreen[LElbow].Y), cvPoint(JointsScreen[LHand].X, JointsScreen[LHand].Y), CV_RGB(0,255,0), 3, 8, 0);
					circle(disparityMap, Point(JointsScreen[RHand].X, JointsScreen[RHand].Y), 20, cvScalar(0));
				}

				flip(disparityMap, flipped, 1); //flip horizontal

				if(mUserGenerator.GetNumberOfUsers() > 0 && RLength && LLength){
					char temp[10];
					sprintf(temp,"%f",JointsScreen[RHand].Z);
					CvFont Font;
					cvInitFont( &Font,CV_FONT_HERSHEY_SIMPLEX,0.5,0.5,0.0,1, CV_AA );
					cvPutText(&(flipped.operator IplImage()), temp, cvPoint(20, 20), &Font, CV_RGB(255,0,0));
					sprintf(temp,"%d", thresholdRHand);
					cvPutText(&(flipped.operator IplImage()), temp, cvPoint(20, 100), &Font, CV_RGB(255,0,0));
				}

				imshow( "colorized disparity map", flipped );

			}

			if( retrievedImageFlags[2] && capture.retrieve( validDepthMap, CV_CAP_OPENNI_VALID_DEPTH_MASK ) )
			{
				//imshow( "valid depth mask", validDepthMap );
			}
			if( retrievedImageFlags[3] && capture.retrieve( bgrImage, CV_CAP_OPENNI_BGR_IMAGE ) )
			{
				//imshow( "rgb image", bgrImage );
			}

			if( retrievedImageFlags[4] && capture.retrieve( grayImage, CV_CAP_OPENNI_GRAY_IMAGE ) )
			{
				//	imshow( "gray image", grayImage );
			}
		}

		if( waitKey( 30 ) >= 0 )
			break;

	}

	return 0;
}

