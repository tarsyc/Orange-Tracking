#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class Position 
{
public:
    vector<Point> historical_points; // 存入的历史识别点
    vector<Point> predicted_points;
    vector<vector<Point>> real_points; // 真实识别点
    Point2f center;
    Point2f predict_point;
    struct centernow
    {
        double x;
        double y;
    };
    struct centernow center_now[1000]; // 保存历史预测点
    float r = 0;
    int i = 0;
    void kalmanPrediction(double time_interval, Mat& frame) 
    {
        const int state_size = 4;
        const int measurement_size = 2;
        KalmanFilter kf(state_size, measurement_size, 0, CV_64F);

        kf.transitionMatrix = (Mat_<double>(state_size, state_size) << 1, 0, time_interval, 0,
                               0, 1, 0, time_interval,
                               0, 0, 1, 0,
                               0, 0, 0, 1);
        kf.measurementMatrix = (Mat_<double>(measurement_size, state_size) << 1, 0, 0, 0,
                                0, 1, 0, 0);

        kf.processNoiseCov = (Mat_<double>(state_size, state_size) << 1e-1, 0, 0, 0,
                               0, 1e-1, 0, 0,
                               0, 0, 1e-1, 0,
                               0, 0, 0, 1e-1);

        kf.measurementNoiseCov = (Mat_<double>(measurement_size, measurement_size) << 1e-5, 0,
                                   0, 1e-5);

        kf.statePost.at<double>(0) = center.x;
        kf.statePost.at<double>(1) = center.y;
        kf.statePost.at<double>(2) = 0;
        kf.statePost.at<double>(3) = 0;

        for (const Point& pos : historical_points) 
        {
            Mat measurement = (Mat_<double>(measurement_size, 1) << pos.x, pos.y);
            kf.correct(measurement);
            kf.statePost = kf.predict();
        }
        Mat prediction = kf.predict();

        predict_point.x = prediction.at<double>(0);
        center_now[i].x = predict_point.x;
        predict_point.y = prediction.at<double>(1);
        center_now[i].y = predict_point.y;
        circle(frame, predict_point, 3, Scalar(0,255, 0), 3);
        circle(frame, predict_point, r, Scalar(0, 255, 0), 3);
        line(frame, center, predict_point, Scalar(0, 255, 255), 3);
        if (i >= 2) 
        {
            Point2f prev_predict_point(center_now[i - 2].x, center_now[i - 2].y);
            circle(frame, prev_predict_point, 3, Scalar(255, 0, 0), 3);
        }

        i++;
    }
};

class Recognize 
{
public:
    void recognize(Mat& frame, Position& position) 
    {
        Mat hsv;
        cvtColor(frame, hsv, COLOR_BGR2HSV);
        Mat mask;
        inRange(hsv, Scalar(10, 100, 100), Scalar(30, 255, 255), mask);
        findContours(mask, position.real_points, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (position.real_points.size() > 0) 
        {
            int maxs = 0;
            int maxl = 0;
            for (int i = 0; i < position.real_points.size(); i++) 
            {
                double s = contourArea(position.real_points[i]);
                if (s > maxs) {
                    maxs = s;
                    maxl = i;
                }
            }
            if (maxl != 0) 
            {
                Point2f center;
                minEnclosingCircle(position.real_points[maxl], center, position.r);
                circle(frame, center, 3, Scalar(0, 0, 255), 3);
                circle(frame, center, position.r, Scalar(255, 255, 255), 3);
                position.center = center;
                position.historical_points.push_back(center);
            }
        }
    }
};

class Direction 
{
public:
    void direction_judgment(Mat& frame, Point2f center, Point2f predict_point) 
    {
        if (predict_point.x-center.x>10) 
        {
            cout << "right" << endl;
            putText(frame, "right", Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
        else if (predict_point.x-center.x<-10) 
        {
            cout << "left" << endl;
            putText(frame, "left",Point(50,100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        }
    }
};

int main() 
{
    VideoCapture capture("/home/czhongyang/orange/orange1.mp4");///home/czhongyang/orange/orange2.mp4
    Mat frame;
    Position pos;
    Recognize recognizer;
    Direction dir;
    double time_interval = 1;

    while (1) 
    {
        capture >> frame;
        if (frame.empty())
        {break;}
        pos.predict_point = pos.center;
        recognizer.recognize(frame, pos);
        pos.kalmanPrediction(time_interval,frame);
        dir.direction_judgment(frame, pos.center, pos.predict_point);
        putText(frame, "Red point: current recognition point", Point(0, 500), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
        putText(frame, "Green point: 1s prediction point", Point(0, 530), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
        putText(frame, "Blue point: historical prediction point", Point(0, 560), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

        imshow("frame", frame);
        if (waitKey(66) == 27)//66ms，每秒15帧
            break;
    }
    capture.release();
    destroyAllWindows();
    return 0;
}
