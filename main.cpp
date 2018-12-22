#include <opencv2/opencv.hpp>  
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define DEBUG

using namespace cv;
using namespace std;

const int imageWidth = 1280;                             //摄像头的分辨率  
const int imageHeight = 480;

bool checkEllipseShape(Mat src,vector<Point> contour,RotatedRect ellipse,double ratio=0.01)
{
	//get all the point on the ellipse point
	vector<Point> ellipse_point;

	//get the parameter of the ellipse
	Point2f center = ellipse.center;
	double a_2 = pow(ellipse.size.width*0.5,2);
	double b_2 = pow(ellipse.size.height*0.5,2);
	double ellipse_angle = (ellipse.angle*3.1415926)/180;
	

	//the uppart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = -ellipse.size.width*0.5+i;
		double y_left = sqrt( (1 - (x*x/a_2))*b_2 );

		//rotate
        //[ cos(seta) sin(seta)]
        //[-sin(seta) cos(seta)]
        cv::Point2f rotate_point_left;
        rotate_point_left.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_left;
        rotate_point_left.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_left;

		//trans
		rotate_point_left += center;

		//store
		ellipse_point.push_back(Point(rotate_point_left));
	}
	//the downpart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = ellipse.size.width*0.5-i;
		double y_right = -sqrt( (1 - (x*x/a_2))*b_2 );

		//rotate
        //[ cos(seta) sin(seta)]
        //[-sin(seta) cos(seta)]
        cv::Point2f rotate_point_right;
		rotate_point_right.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_right;
        rotate_point_right.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_right;

		//trans
		rotate_point_right += center;

		//store
		ellipse_point.push_back(Point(rotate_point_right));

	}


	//vector<vector<Point> > contours1;
	//contours1.push_back(ellipse_point);
	//drawContours(img1,contours1,-1,Scalar(255,0,0),2);

	//match shape
	double a0 = matchShapes(ellipse_point,contour,CV_CONTOURS_MATCH_I1,0);  
	if (a0>0.01)
	{
		return true;      
	}

	return false;
}

void Coutour2EllipsePoints(Mat src, vector<vector<Point> > contours, vector<Point2f>& img_ellipse_points)
{
	//fit ellipse
	vector<RotatedRect> minEllipse(contours.size());
	vector<Point> ellipse_points;
  	for( int i = 0; i < contours.size(); i++ )
	{ 
		//point size check
		if(contours[i].size()<20)
		{
			continue;
		}

		//point area
		if(contourArea(contours[i])<200)
		{
			continue;
		}

		minEllipse[i] = fitEllipse(Mat(contours[i]));

		//check shape
		if(checkEllipseShape(src,contours[i],minEllipse[i]))
		{
			continue;
		}

		ellipse_points.insert(ellipse_points.end(), contours[i].begin(), contours[i].end());
		//ellipse( src, minEllipse[i], Scalar( 0, 0, 255), 2);
	}

	//convert Point to Point2f
	for(auto point:ellipse_points)
	{
		img_ellipse_points.push_back(Point2f((float)point.x, (float)point.y));
	}
}

/** @function main */
int main( int argc, char** argv )
{	
	//读取内部参数
	FileStorage fs("../stereoParams.yml", FileStorage::READ);
	if(!fs.isOpened())
	{
	      cout << "Open param files fail!" << endl;
	      return -1;
	}
	
	int channel, times;
	fs["channel"] >> channel;
	fs["times"] >> times;

	Mat M1, D1, M2, D2, R, T;
	fs["M1"] >> M1;
	fs["D1"] >> D1;
	fs["M2"] >> M2;
	fs["D2"] >> D2;
	fs["R"] >> R;
	fs["T"] >> T;
	
	Mat Rl, Pl, Rr, Pr, Q;
	Rect roi1, roi2;
	Size img_size = Size(imageWidth /2, imageHeight);

	/*
	立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
	使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，
	这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
	stereoRectify 这个函数计算的就是从图像平面投影都公共成像平面的旋转矩阵Rl,Rr。 
	Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
	左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
	其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w] 
	Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。
	其中d为左右两幅图像的时差
	*/
	//Alpha取值为-1时，保持默认值;Alpha取值为0时，裁剪图像
	stereoRectify(M1, D1, M2, D2, img_size, R, T, Rl, Rr, Pl, Pr, Q,
		CALIB_ZERO_DISPARITY, 0, img_size, &roi1, &roi2);
	
	#ifdef DEBUG
	cout << "roi1 x: " << roi1.x << " ,y: " << roi1.y << endl;
	cout << "roi2 x: " << roi2.x << " ,y: " << roi2.y << endl;
	#endif
	cout << "Q: " << endl << Q << endl;
	
	const double base_line = 1.0 / Q.at<double>(3,2);
	const double cx_l = -Q.at<double>(0,3);
	const double cy_l = -Q.at<double>(1,3);
	const double focus = Q.at<double>(2,3);
	
	/*
	根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
	mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
	ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。
	在openCV里面，校正后的摄像机矩阵Mrect是跟投影矩阵P一起返回的。
	所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
	*/
	//获取两相机的矫正映射
	Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, Rl, Pl, img_size, CV_16SC2, map11, map12);
	initUndistortRectifyMap(M2, D2, Rr, Pr, img_size, CV_16SC2, map21, map22);
	
	VideoCapture cap(channel);
	if(argc == 1)
	{
	    cout << "测试两个摄像头同时读取数据" << endl;

	    if (!cap.isOpened())
	    {
		    cout << "Open camera fail!" << endl;
		    return -1;
	    }
	    cap.set(CV_CAP_PROP_FRAME_WIDTH, imageWidth);
	    cap.set(CV_CAP_PROP_FRAME_HEIGHT, imageHeight);
	}
	else
	{
	    cap.release();
	}
	
	bool stop = false;
	Mat frame, img1, img2;
	
	int count = 1;
	
	while(!stop)
	{
	      if(++count > times)
	      {
		    stop = true;
	      }
	      
	      	      
	      if(argc == 1)
	      {
		  cap >> frame;
		  img1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
		  img2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

		  //Mat img_remap_1, img_remap_2;
		  remap(img1, img1, map11, map12, INTER_LINEAR);
		  remap(img2, img2, map21, map22, INTER_LINEAR);
		  //img1 = img_remap_1.clone();
		  //img2 = img_remap_2.clone();
	      }
	      else if(argc == 2)
	      {
		  frame = imread(argv[1], 1);
		  img1 = frame(Rect(0, 0, frame.cols / 2, frame.rows));
		  img2 = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));
	      }
	      else if(argc == 3)
	      {
		
		  //load images
		  img1 = imread( argv[1], 1 );
		  img2 = imread( argv[2], 1 );
	      }
	      else
	      {
		  cout << "Input Error!" << endl;
		  return -1;
	      }
	      
	      // convert into gray
	      Mat gray1, gray2;
	      cvtColor( img1, gray1, CV_BGR2GRAY );
	      cvtColor( img2, gray2, CV_BGR2GRAY );
	      
	      // find contours
	      vector<vector<Point> > contours1, contours2;
	      int thresh = threshold( gray1, gray1, 0, 255, CV_THRESH_BINARY|CV_THRESH_OTSU);
	      threshold( gray2, gray2, thresh, 255, CV_THRESH_BINARY);
	      findContours( gray1, contours1, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	      findContours( gray2, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
	      
	      //drawContours( img1, contours1, -1, Scalar(0, 0, 255), 2);
	      //drawContours( img2, contours2, -1, Scalar(0, 0, 255), 2);
	      #ifdef DEBUG
	      namedWindow("gray1", 0);
	      namedWindow("gray2", 0);
	      imshow("gray1", gray1);
	      imshow("gray2", gray2);
	      #endif
	      
	      if( !(contours1.size() && contours2.size()) )
	      {
		    #ifdef DEBUG
		    if( waitKey(10) == 'q' )
		    {
			  stop = true;
		    }
		    #endif
		    
		    continue;		
	      }
	      
	      vector<Point2f> img1_ellipse_points, img2_ellipse_points;
	      Coutour2EllipsePoints(img1, contours1, img1_ellipse_points);
	      Coutour2EllipsePoints(img2, contours2, img2_ellipse_points);
	      #ifdef DEBUG
	      cout << "img1_ellipse_points: " << img1_ellipse_points.size() << ", img2_ellipse_points: " << img2_ellipse_points.size() << endl;
	      #endif
	      
	      if( !(img1_ellipse_points.size() && img2_ellipse_points.size()) )
	      {
		    #ifdef DEBUG
		    if( waitKey(10) == 'q' )
		    {
			  stop = true;
		    }
		    #endif
		    
		    continue;
	      }
	      
	      vector<KeyPoint> keypoints_1, keypoints_2;
	      KeyPoint::convert(img1_ellipse_points, keypoints_1);
	      KeyPoint::convert(img2_ellipse_points, keypoints_2);

	      Mat descriptors_1, descriptors_2;
	      Ptr<ORB> orb = ORB::create(1000);
	      orb->compute(img1, keypoints_1, descriptors_1);
	      orb->compute(img2, keypoints_2, descriptors_2);
	      
	      if(descriptors_1.cols != descriptors_2.cols)
	      {
		    cout << "Error!" << endl;
		    continue;
	      }

	      //Mat outimg1, outimg2;
	      //drawKeypoints(img1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	      //drawKeypoints(img2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	      /// 结果在窗体中显示
	      //imwrite("ellipse.jpg",img1);
	      //namedWindow("ellipse_mark_1", 0);
	      //namedWindow("ellipse_mark_2", 0);
	      //imshow("ellipse_mark_1", outimg1);
	      //imshow("ellipse_mark_2", outimg2);

	      vector<DMatch> matches;
	      BFMatcher matcher(NORM_HAMMING);
	      matcher.match(descriptors_1, descriptors_2, matches);
	      
	      if(matches.size() == 0)
	      {
		    continue;
	      }
	      
	      double min_dist = 10000, max_dist = 0;
	      for(int i = 0; i < descriptors_1.rows; i++)
	      {
		    double dist = matches[i].distance;
		    if(dist < min_dist) min_dist = dist;
		    if(dist > max_dist) max_dist = dist;		    
	      }
	      #ifdef DEBUG
	      cout << "Max dist: " << max_dist << ", Min dist: " << min_dist << endl;
	      #endif
	      
	      vector<DMatch> good_matches;
	      for(int i = 0; i < descriptors_1.rows; i++)
	      {
		    if(matches[i].distance <= min(max(2*min_dist, 15.0), 30.0))
		    {
			  good_matches.push_back(matches[i]);
		    }
	      }

	      #ifdef DEBUG
	      Mat img_match;
	      drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, img_match);
	      namedWindow("img_match", 0);
	      imshow("img_match", img_match);
	      
	      cout << "good matches size: " << good_matches.size() << endl;
	      for(auto good_match:good_matches)
	      {
		    cout << "A pair of points: " << keypoints_1[good_match.queryIdx].pt << ", " 
			  << keypoints_2[good_match.trainIdx].pt << endl;
	      }
	      
	      if(matches.size())
	      {
		    if( waitKey(50) == 'q' )
		    {
			  stop = true;
		    }
	      }
	      else
	      {
		    if( waitKey(10) == 'q' )
		    {
			  stop = true;
		    }
	      }
	      #endif
	      
	      if(good_matches.size() > 3)
	      {
		    //imwrite("../data/"+to_string(count)+".jpg", frame);
		    vector<Point3d> good_world_points;
		    for(auto good_match:good_matches)
		    {
			  Point2f left_point = keypoints_1[good_match.queryIdx].pt;
			  Point2f right_point = keypoints_2[good_match.trainIdx].pt;
			  if(fabs(left_point.y - right_point.y) <= 3)
			  {
				float disparity = left_point.x - right_point.x;
				Point3d world_point;
				world_point.x = (left_point.x - cx_l) * float(base_line) / disparity;
				world_point.y = (left_point.y - cy_l) * float(base_line) / disparity;
				world_point.z = focus * float(base_line) / disparity;
				good_world_points.push_back(world_point);
			  }
		    }
		    
		    //init eigen matrix and calculate the mean point
		    int good_world_points_size = good_world_points.size();
		    Eigen::MatrixXd matrix_x3 = Eigen::MatrixXd::Random(good_world_points_size, 3);
		    Eigen::Vector3d centroid;
		    for(int i = 0; i < good_world_points_size; i++)
		    {
			  matrix_x3(i,0) = good_world_points[i].x;
			  matrix_x3(i,1) = good_world_points[i].y;
			  matrix_x3(i,2) = good_world_points[i].z;
			  centroid(0,0) += matrix_x3(i,0);
			  centroid(1,0) += matrix_x3(i,1);
			  centroid(2,0) += matrix_x3(i,2);
		    }
		    centroid /= good_world_points_size;
		    cout << "centroid point: " << endl << centroid << endl;
		    
		    for(int i = 0; i < good_world_points_size; i++)
		    {
			  matrix_x3.row(i) -= centroid.transpose();
		    }
		    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix_x3, Eigen::ComputeFullU | Eigen::ComputeFullV);
		    Eigen::MatrixXd V = svd.matrixV();
		    cout << "V: " << endl << V << endl;
		    
		    Eigen::Vector4f plane_coeff;
		    //AX+BY+CZ+D=0
		    plane_coeff(0,0) = V(0,2);  //A
		    plane_coeff(1,0) = V(1,2);  //B
		    plane_coeff(2,0) = V(2,2);  //C
		    plane_coeff(3,0) -= plane_coeff(0,0)*centroid(0,0)+
					plane_coeff(1,0)*centroid(1,0)+plane_coeff(2,0)*centroid(2,0);  //D
		    cout << "plane_coeff: " << endl << plane_coeff << endl;
		    
		    const float &A = plane_coeff(0,0), &B = plane_coeff(1,0);
		    const float &C = plane_coeff(2,0), &D = plane_coeff(3,0);
		    const float &X = centroid(0,0), &Y = centroid(1,0), &Z = centroid(2,0);
		    
		    //坐标原点到平面的距离，即相机高度
		    double camera_height = fabs(D)/sqrt(A*A+B*B+C*C);
		    cout << "camera_height: " << endl << camera_height << endl;
		    double horizon_distance = sqrt(Z*Z-camera_height*camera_height);
		    cout << "horizon_distance: " << endl << horizon_distance << endl;
	      }
	}
	
	fs.release();
	cap.release();
  	return 1;
}
