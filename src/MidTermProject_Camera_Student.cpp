/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    //method
    string detectorType = "HARRIS"; //HARRIS, SHITOMASI, FAST, BRISK, ORB, AKAZE, SIFT
    string descriptorType = "BRISK"; // BRIEF, BRISK, ORB, FREAK, AKAZE, SIFT
    /* MATCH KEYPOINT DESCRIPTORS */
    string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
    string matchDescriptorType; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN   

    //vector of keypoints number
    vector<int> kptsNumber;
    vector<int> matchNumber;

    //Time of detection and extraction
    double detTime = 0;
    double descTime = 0;
    

    if (descriptorType.compare("BRIEF") == 0 || descriptorType.compare("BRISK") == 0) matchDescriptorType = "DES_BINARY";
    else matchDescriptorType = "DES_HOG";
    

    /* Crop keypoints except that are on the preceding vehicle */
    bool bFocusOnVehicle = true;

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame>dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;

        if ( dataBuffer.size() < dataBufferSize )
        {
            dataBuffer.push_back(frame);
        }
        else
        {
            dataBuffer.erase(dataBuffer.begin());
            dataBuffer.push_back(frame);
        }

        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, detTime, bVis);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, detTime, bVis);          
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, detTime, bVis);
        }
        bVis = false;

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        cv::Rect vehicleRect(535, 180, 180, 150);
        for (auto it = keypoints.begin(); it != keypoints.end() ; ++it)
        {
            if( !vehicleRect.contains(it->pt))
            {
                keypoints.erase(it);
                --it;
            }
        }
        //// EOF STUDENT ASSIGNMENT

        /////////////MP7////////////////////////
        kptsNumber.push_back(keypoints.size());

        // optional : limit number of keypoints (helpful for debugging and learning)
        // bool bLimitKpts = true;
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 100;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        cv::Mat descriptors;
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, descTime);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            vector<cv::DMatch> matches;
            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, matchDescriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            /////////////MP7////////////////////////
            matchNumber.push_back(matches.size());

            // visualize matches between current and previous image
            // bVis = true;
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }

    } // eof loop over all images

    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + 0;
    string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
    cv::Mat img, imgGray;
    img = cv::imread(imgFullFilename);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, (dataBuffer.end() - 1)->keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "kpts";
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, visImage);
    cout << "Press key to continue to next image" << endl;
    cv::waitKey(0); // wait for key to be pressed
    std::string name = detectorType+std::string("image.png");
    cv::imwrite(name, visImage);


    /////////////////////MP7///////////////////
    float kptsNumSum = 0.0;
    cout << "Number of Keypoints " << detectorType << endl;
    for (auto it = kptsNumber.begin() ; it != kptsNumber.end(); ++it) 
    {
        cout << *it << ", ";
        kptsNumSum += *it;
    }
    cout << endl; 

    /////////////////////MP8///////////////////
    float matchNumSum = 0.0;
    cout << "Number of Matched " << detectorType << " & " << descriptorType << endl;
    for (auto it = matchNumber.begin() ; it != matchNumber.end(); ++it) 
    {
        cout << *it << ", ";
        matchNumSum += *it;
    }
    cout << endl;
    cout << "Number of Average Matched  " << matchNumSum/matchNumber.size() << endl;
    cout << detectorType << "  Detection time " << (1000 * detTime)/ (1.0 * (imgEndIndex - imgStartIndex + 1)) << " ms"<< endl;
    cout << descriptorType << "  Extraction time " << (1000 * descTime)/ (1.0 * (imgEndIndex - imgStartIndex + 1)) << " ms"<< endl;

    return 0;
}
