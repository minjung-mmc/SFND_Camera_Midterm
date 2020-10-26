# SFND_2D_Feature_Matching

## MP.1 Data Buffer Optimization
Set up the loading procedure for the images. Make Data Buffer, by deleting the oldest one from one end of the vector and the adding the new one to the other end.
```
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
```

## MP.2 Keypoint Detection
Implement a selection of alternative detectors, HARRIS, Shi-Tomasi, FAST, BRISK, ORB and AKAZE by setting the string 'detectorType', and detect keypoints.

### Harris Detector
```
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double &detTime, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    //get time started
    double t = (double)cv::getTickCount();
      
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT); //apply corner Harris
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());


    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response; //create a new keypoint .. 

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it); //overlap keytpoint with new keypoint (which is certain points that is over threshold)
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)  // if overlap is >t AND response is higher for new kpt
                        {                     
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "HARRIS detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    detTime += t;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "HARRIS Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}
```
### FAST, BRISK, ORB, AKAZE, SIFT Detector
```
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, double &detTime, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0)
    {
    // FAST detector
    int threshold = 30; //difference between intensity of the central pixel and pixels of a circle around this pixel
    bool BNMS = true; //perform non-maxima suppression on keypoints
    cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16;
    detector = cv::FastFeatureDetector::create(threshold, BNMS, type);
    }

    else if (detectorType.compare("BRISK") == 0)
    {
        //BRISK detector
        int threshold = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        detector = cv::BRISK::create(threshold, octaves, patternScale);
     }


    else if (detectorType.compare("ORB") == 0)
    {
        detector = cv::ORB::create();
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        detector = cv::AKAZE::create();
    }
    else if (detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
    }
    else
    {
        cout << " You entered wrong detector type" << endl;
    }

    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " with n = " << keypoints.size() << "keypoints in" << 1000 * t / 1.0 << " ms" << endl;
    detTime += t;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + "Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}
```

## MP.3 Keypoint Removal
Remove all keypoints outside of a bounding box around the preceding vehicle. Use 'React' datatype to do this. Box parameters: cx = 535, cy = 180, w = 180, h = 150
```
cv::Rect vehicleRect(535, 180, 180, 150);
for (auto it = keypoints.begin(); it != keypoints.end() ; ++it)
{
    if( !vehicleRect.contains(it->pt))
    {
        keypoints.erase(it);
        --it;
    }
}
```

## MP.4 Keypoint Descriptors
Implement a selection of alternative descriptors, BRIEF, ORB, FREAK, AKAZE and SIFT by setting the string 'descriptorType', and descript keypoints.

```
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &descTime)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        extractor = cv::BRISK::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();   
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();   
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();   
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else
    {
        cout << "You entered wrong descriptor type" << endl;
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl; 
    descTime += t;
}
```

## MP.5 Descriptor Matching
Implement FLANN matching as well as k-nearest neighbor selection.
```
bool crossCheck = false;
cv::Ptr<cv::DescriptorMatcher> matcher;
    
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
            cout << "Converted to CV_32F" << endl;
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }

    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // parameters
        vector<vector<cv::DMatch>> knn_matches;
        auto k = 2;
        double minDescDistRatio = 0.8;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) 
            //if best match SSD is better than 0.8*second best match SSD, then push_back.
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}
```

## MP.6 Descriptor Distance Ratio
implement the descriptor distance ratio test as a filtering method to remove bad keypoint matches.

```
vector<vector<cv::DMatch>> knn_matches;
auto k = 2;
double minDescDistRatio = 0.8;
matcher->knnMatch(descSource, descRef, knn_matches, k);

for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
{
    if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) 
    //if best match SSD is better than 0.8*second best match SSD, then push_back.
    {
        matches.push_back((*it)[0]);
    }
}
```

## MP.7 Performance Evaluation 1
 Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented. 

### Number of Keypoints

|Detector       | Keypoint #(10 frames - ROI)	 	 	 | Keypoint #(Average - ROI)|
| ------------- | ---------------------------------------------	 |:---------------------:   |
| Harris        |17, 14, 18, 21, 26, 43, 18, 31, 26, 34 	 |24.80 		    |
| Shi-Tomasi	|125, 118, 123, 120, 120, 113, 114, 123, 111 ,112|117.90      		    |
| FAST	        |149, 152, 150, 155, 149, 149, 156, 150, 138, 143|149.10      		    |
| BRISK		|264, 282, 282, 277, 297, 279, 289, 272, 266, 254|276.20       		    |
| ORB		|92, 102, 106, 113, 109, 125, 130, 129, 127, 128 |116.10       		    |
| AKAZE		|166, 157, 161, 155, 163, 164, 173, 175, 177, 179|167.00      		    |
| SIFT    	|138, 132, 124, 137, 134, 140, 137, 148, 159, 137|138.60       		    |


### Neighbourhood Size

- Harris, Shi-Tomasi and FAST has small neighborhood size. Also, size of neighbothood is fixed.
- AKAZE has medium neighborhood size.
- BRISK, ORB and SIFT has large neighborhood size.

If the distribution of nearborhood size is large, detector is robust to the scale change and rotation. Therefore, it can be said that detector which has large neighborhood size like BRISK, ORB and SIFT, performs well with zoomed in or out pictures, as well as rotated ones.

1. HARRIS
![HARRIS](./neighbourSize/HARRIS.png)


2. SHITOMASI
![SHITOMASI](./neighbourSize/SHITOMASI.png)


3. FAST
![FAST](./neighbourSize/FAST.png)


4. BRISK
![BRISK](./neighbourSize/BRISK.png)


5. ORB
![ORB](./neighbourSize/ORB.png)


6. AKAZE
![AKAZE](./neighbourSize/AKAZE.png)


7. SIFT
![SIFT](./neighbourSize/SIFT.png)


## MP.8 Performance Evaluation 2
 Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, use the BF approach with the descriptor distance ratio set to 0.8.

**AKAZE descriptor only worked with AKAZE detector **
**ORB descriptor did not work with SIFT detector **

### Number of Matched Keypoints


|Detector/Descriptor       | Matched #(9 Matches - ROI)	 	 	 | Matched #(Average - ROI)|
| ------------- | ---------------------------------------------	 |:---------------------:   |
| **Harris**    |	 					 | 		            |
|  +BRISK       | 12, 10, 14, 15, 16, 16, 15, 23, 21		 |15.78 		    |
|  +BRIEF       | 14, 11, 15, 20, 24, 26, 16, 24, 23	 	 |19.22		            |
|  +ORB         | 11, 11, 15, 17, 20, 18, 12, 21, 20	         |16.11			    |
|  +FREAK       | 11, 9, 13, 14, 13, 18, 10, 17, 18     	 |13.67			    |
|  +SIFT        | 14, 11, 16, 19, 22, 22, 13, 24, 22     	 |18.11			    |
|**Shi-Tomasi**	|						 |      		    |
|  +BRISK       | 95, 88, 80, 90, 82, 79, 85, 86, 82	 	 |85.22 		    |
|  +BRIEF       | 115, 111, 104, 101, 102, 102, 100, 109, 100 	 |104.89 		    |
|  +ORB         | 87, 82, 87, 90, 83, 77, 84, 86, 89		 |85.00 		    |
|  +FREAK       | 66, 66, 64, 63, 62, 64, 61, 65, 63	 	 |63.78			    |
|  +SIFT        | 112, 109, 104, 103, 99, 101, 96, 106, 97 	 |103.00		    |
| **FAST**      |						 |	    		    |
|  +BRISK       | 97, 104, 101, 98, 85, 107, 107, 100, 100 	 |99.89 		    |
|  +BRIEF       | 119, 130, 118, 126, 108, 123, 131, 125, 119 	 |122.11	            |
|  +ORB         | 95, 102, 87, 94, 88, 95, 104, 96, 98		 |95.44		            |
|  +FREAK       | 64, 80, 65, 79, 61, 76, 83, 77, 82     	 |74.11                     |
|  +SIFT        | 118, 123, 110, 119, 114, 119, 123, 117, 103  	 |116.22                    |
| **BRISK**	|						 |	     		    |
|  +BRISK       | 171, 176, 157, 176, 174, 188, 173, 171, 184	 |174.44 		    |
|  +BRIEF       | 178, 205, 185, 179, 183, 195, 207, 189, 183 	 |189.33 		    |
|  +ORB         | 97, 103, 91, 92, 82, 117, 109, 108, 114 	 |101.44 	            |
|  +FREAK       | 114, 121, 113, 118, 103, 129, 135, 129, 131 	 |121.44 	            |
|  +SIFT        | 182, 193, 169, 183, 171, 195, 194, 176, 183 	 |182.89 	            |
| **ORB**	|						 |      		    |
|  +BRISK       | 73, 74, 79, 85, 79, 92, 90, 88, 91	 	 |83.44 	            |
|  +BRIEF       | 49, 43, 45, 59, 53, 78, 68, 84, 66		 |60.56			    |
|  +ORB         | 40, 52, 46, 54, 53, 65, 68, 65, 72	 	 |57.22 		    |
|  +FREAK       | 39, 33, 37, 40, 33, 40, 41, 39, 44	 	 |38.44 		    |
|  +SIFT        | 67, 79, 78, 79, 82, 95, 95, 94, 94	 	 |84.78 		    |
| **AKAZE**	|						 |      		    |
|  +BRISK       | 137, 125, 129, 129, 131, 132, 142, 146, 144	 |135.00 		    |
|  +BRIEF       | 141, 134, 131, 130, 134, 146, 150, 148, 152 	 |140.67	            |
|  +ORB         | 102, 91, 97, 86, 95, 114, 107, 112, 118	 |102.44 		    |
|  +FREAK       | 103, 105, 93, 99, 97, 115, 126, 118, 117 	 |108.11 	            |
|  +AKAZE       | 128, 128, 125, 117, 121, 132, 137, 140, 144 	 |130.22 	            |
|  +SIFE        | 134, 134, 130, 136, 137, 147, 147, 154, 151 	 |141.11                    |
| **SIFT**    	|						 |      		    |
|  +BRISK       | 64, 66, 62, 66, 59, 64, 64, 67, 80		 |65.78 		    |
|  +BRIEF       | 86, 78, 76, 85, 69, 74, 76, 70, 88 		 |78.00		            |
|  +FREAK       | 59, 63, 54, 64, 51, 50, 47, 53, 65 		 |56.22 	            |
|  +SIFT        | 82, 81, 85, 93, 90, 81, 82, 102, 104  	 |88.89 	            |


## MP.9 Performance Evaluation 3
Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, suggest the TOP3 detector / descriptor combinations as the best choice for our purpose of detecting keypoints on vehicles.

### Computation Time

|Detector/Descriptor| Detection Time(ms)	 	 	 |Extraction Time(ms)	    |
| ------------- | :-------------------------------------------:	 |:---------------------:   |
| **Harris**    |	 		13.84			 | 		            |
|  +BRISK       | 						 |0.79   		    |
|  +BRIEF       |  						 |0.70 			    |
|  +ORB         |					  	 |2.65		            |
|  +FREAK       |  						 |29.91	 		    |
|  +SIFT        |  						 |10.64	 		    |
|**Shi-Tomasi**	|			11.46			 |      		    |
|  +BRISK       | 						 |1.45		            |
|  +BRIEF       |  						 |0.96		            |
|  +ORB         |					  	 |2.80 		            |
|  +FREAK       |  						 |29.01		            |
|  +SIFT        |  						 |14.72		            |
| **FAST**      |			0.96			 |	    		    |
|  +BRISK       | 						 |1.39		            |
|  +BRIEF       |  						 |0.71		            |
|  +ORB         |					  	 |3.25		            |
|  +FREAK       |  						 |28.69		            |
|  +SIFT        |  						 |17.83		            |
| **BRISK**	|			31.57			 |	     		    |
|  +BRISK       | 						 |2.29		            |
|  +BRIEF       |  						 |0.87		            |
|  +ORB         |					  	 |10.30		            |
|  +FREAK       |  						 |29.50		            |
|  +SIFT        |  						 |21.50		            |
| **ORB**	|			10.57			 |      		    |
|  +BRISK       | 						 |1.08		            |
|  +BRIEF       |  						 |0.55		            |
|  +ORB         |					  	 |11.27		            |
|  +FREAK       |  						 |29.73		            |
|  +SIFT        |  						 |29.73		            |
| **AKAZE**	|			45.66			 |      		    |
|  +BRISK       | 						 |1.38		            |
|  +BRIEF       |  						 |0.58		            |
|  +ORB         |					  	 |7.27		            |
|  +FREAK       |  						 |29.47		            |
|  +AKAZE       | 			                 	 |37.23			    |
|  +SIFT        | 			                 	 |15.65			    |
| **SIFT**    	|			77.1			 |      		    |
|  +BRISK       | 				 	 	 |1.17		            |
|  +BRIEF       | 				 	 	 |0.56		            |
|  +FREAK       | 				 	 	 |29.13		            |
|  +SIFT        | 	`			 	 	 |55.17		            |



### Best Detector/Descriptor Combination

**Standard**

1. Run Time 
To run algorithm real-time computation time will be important.

   **Best Performance : FAST + BRIEF**

2. Matching rate(# of Matched Key Point/# of Key Point)
For algorithms to be robust, the combination of the detector and the discriptor is important. I considered the performance of the combination as a percentage of the key points that was matched.

   **Best Performance : Shi-Tomasi + BRIEF**

    |Detector/Descriptor|Matching Rate | 
    | ------------- | -------------- |
    | **Harris**    |	 	 |
    |  +BRISK       |0.64 		 |
    |  +BRIEF       |0.78	 	 |
    |  +ORB         |0.65 	         |
    |  +FREAK       |0.55		 |
    |  +SIFT        |0.73		 |
    |**Shi-Tomasi**	|		 |
    |  +BRISK       |0.72	 	 |
    |  +BRIEF       |0.89		 |
    |  +ORB         |0.72		 |
    |  +FREAK       |0.54		 |
    |  +SIFT        |0.87		 |
    | **FAST**      |		 |
    |  +BRISK       | 0.67 		 |
    |  +BRIEF       | 0.82	 	 |
    |  +ORB         | 0.64		 |
    |  +FREAK       | 0.50 		 |
    |  +SIFT        | 0.78 		 |
    | **BRISK**	|		 |
    |  +BRISK       | 0.63 		 |
    |  +BRIEF       | 0.69		 |
    |  +ORB         | 0.37		 |
    |  +FREAK       | 0.44		 |
    |  +SIFT        | 0.66		 |
    | **ORB**	|		 |
    |  +BRISK       | 0.72		 |
    |  +BRIEF       | 0.52		 |
    |  +ORB         | 0.49		 |
    |  +FREAK       | 0.33	 	 |
    |  +SIFT        | 0.73	 	 |
    | **AKAZE**	|		 |
    |  +BRISK       | 0.81		 |
    |  +BRIEF       | 0.84		 |
    |  +ORB         | 0.61		 |
    |  +FREAK       | 0.65	 	 |
    |  +AKAZE       | 0.78 		 |
    |  +SIFT        | 0.84 		 |
    | **SIFT**    	|		 |
    |  +BRISK       | 0.47		 |
    |  +BRIEF       | 0.64		 |
    |  +FREAK       | 0.41		 |
    |  +SIFT        | 0.64		 |
 
   


