#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

void CalibIntrinsicMatrix(string folder, int img_num, Mat &cameraMatrix, Mat &distCoeffs, vector<vector<Point2f> > &imagePoints,  vector<vector<Point3f> > &objectPoints);
void CalibStereoCamera(Mat cameraMatrix1, Mat cameraMatrix2, Mat distCoeffs1, Mat distCoeffs2, vector<vector<Point3f> > &objectPoints, vector<vector<Point2f> > &imagePoints1, vector<vector<Point2f> > &imagePoints2,Mat &R,Mat &T);
/**
 * 
 * 
 * Format :  ./calib folder1 image_num1 [ floder2 image_num2 ] 
 **/
Size image_size;
int main(int argc,char *argv[])
{
    // Monocular Camera Model
    if(argc == 3)
    {
        int img_num;
        sscanf(argv[2],"%d",&img_num);
        Mat cameraMatrix,distCoeffs;
        vector<vector<Point2f> > imagePoints;
        vector<vector<Point3f> > objectPoints;
        CalibIntrinsicMatrix(argv[1],img_num,cameraMatrix,distCoeffs,imagePoints,objectPoints);
        cout << "相机内参数矩阵：" << endl;
        cout << cameraMatrix << endl << endl;
        cout << "畸变系数：\n";
        cout << distCoeffs << endl << endl;

    }
    // Stereo Camera Model
    else if(argc == 4){
        int img_num;
        sscanf(argv[3],"%d",&img_num);
        Mat cameraMatrix1,distCoeffs1;
        Mat cameraMatrix2,distCoeffs2;
        vector<vector<Point2f> > imagePoints1,imagePoints2;
        vector<vector<Point3f> > objectPoints1,objectPoints2;
        CalibIntrinsicMatrix(argv[1],img_num,cameraMatrix1,distCoeffs1,imagePoints1,objectPoints1);
        CalibIntrinsicMatrix(argv[2],img_num,cameraMatrix2,distCoeffs2,imagePoints2,objectPoints2);
        cout << "相机1内参数矩阵：" << endl;
        cout << cameraMatrix1 << endl << endl;
        cout << "畸变系数：\n";
        cout << distCoeffs1 << endl << endl;
        cout<<" ================================= "<<endl;

        cout << "相机2内参数矩阵：" << endl;
        cout << cameraMatrix2 << endl << endl;
        cout << "畸变系数：\n";
        cout << distCoeffs2 << endl << endl;
        cout<<" ================================= "<<endl;

        Mat R,T;
        Mat R1,R2;
        Mat P1,P2;
        Mat Q;
        CalibStereoCamera(cameraMatrix1,cameraMatrix2,distCoeffs1,distCoeffs2,objectPoints1,imagePoints1,imagePoints2,R,T);
        stereoRectify(cameraMatrix1,distCoeffs1,cameraMatrix2,distCoeffs2,image_size,R,T,R1,R2,P1,P2,Q);
        cout << "相机1校正变换矩阵：" << endl;
        cout << R1 << endl << endl;
        cout << "新坐标系下的投影矩阵：\n";
        cout << P1 << endl << endl;
        cout<<" ================================= "<<endl;

        cout << "相机2校正变换矩阵：" << endl;
        cout << R2 << endl << endl;
        cout << "新坐标系下的投影矩阵：\n";
        cout << P2 << endl << endl;
        cout<<" ================================= "<<endl;
    }

}

void CalibIntrinsicMatrix(string folder, int img_num, Mat &cameraMatrix, Mat &distCoeffs, vector<vector<Point2f> > &imagePoints,  vector<vector<Point3f> > &objectPoints)
{
    vector<Mat> imgs;
    for(int i =0;i<img_num;i++)
    {
        stringstream ss;
        ss<<setw(5)<<setfill('0')<<i;
        Mat image = imread(folder+"/"+ss.str()+".png",IMREAD_GRAYSCALE);
        if(!image.data)
        {
            cout<<"Can not find image ..."<<endl;
            return;
        }
        imgs.push_back(image);
    }
    image_size = Size(imgs[0].cols,imgs[0].rows);
    // Set board size 
    Size board_size = Size(11, 9);
    vector<Point2f> image_points_buf;
    vector<vector<Point2f> > imagePoint;
    // 提取角点
    for(int i =0;i<img_num;i++){
        cv::Mat image;
        imgs[i].copyTo(image);
        if (!findChessboardCorners(image, board_size, image_points_buf))
        {
            cout << "Image"<<i<<" : Can not find chessboard corners!"<<endl;
        }
        else
        {
            find4QuadCornerSubpix(image, image_points_buf, Size(5, 5));  
            drawChessboardCorners(image, board_size, image_points_buf, true); 
            imagePoints.push_back(image_points_buf);
            imshow("Camera Calibration", image);  
            waitKey(20);
        }
    }

    // 相机标定
    Size square_size = Size(38, 38);    //标定板格子大小 mm
    for (int t = 0; t<img_num; t++)
    {
        vector<Point3f> tempPointSet;
        for (int i = 0; i<board_size.height; i++)
        {
            for (int j = 0; j<board_size.width; j++)
            {
                Point3f realPoint;
                realPoint.x = i*square_size.width;
                realPoint.y = j*square_size.height;
                realPoint.z = 0;
                tempPointSet.push_back(realPoint);
            }
        }
        objectPoints.push_back(tempPointSet);
    }

    cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
    vector<int> point_counts;
    distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));   //畸变系数：k1,k2,p1,p2,k3
    vector<Mat> tvecsMat;
    vector<Mat> rvecsMat;
    calibrateCamera(objectPoints, imagePoints, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
}

void CalibStereoCamera(Mat cameraMatrix1, Mat cameraMatrix2, Mat distCoeffs1, Mat distCoeffs2, vector<vector<Point3f> > &objectPoints, vector<vector<Point2f> > &imagePoints1, vector<vector<Point2f> > &imagePoints2,Mat &R, Mat &T)
{
    Mat E, F;
    double rms = stereoCalibrate(objectPoints, imagePoints1, imagePoints2,
		cameraMatrix1, distCoeffs1,
		cameraMatrix2, distCoeffs2,
		image_size, R, T, E, F);
    cout << "done with RMS error=" << rms << endl;

    cout << "旋转矩阵：" << endl;
    cout << R << endl << endl;
    cout << "平移矩阵\n";
    cout << T << endl << endl;
}
