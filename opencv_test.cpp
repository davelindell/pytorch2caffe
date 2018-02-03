/**M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

const char* params
    = "{ help           | false | Sample app for loading googlenet model }"
      "{ proto          | pillnet.prototxt | model configuration }"
      "{ model          | pillnet.caffemodel | model weights }"
      "{ label          | synset_words.txt | names of ILSVRC2012 classes }"
      "{ image          | ../pytorch_keras/test.png | path to image file }"
      "{ opencl         | false | enable OpenCL }"
;

float MEAN = 16.861;
float STD = 56.475;


int main(int argc, char **argv)
{
    CV_TRACE_FUNCTION();

    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelTxt = parser.get<string>("proto");
    String modelBin = parser.get<string>("model");
    String imageFile = parser.get<String>("image");
    String classNameFile = parser.get<String>("label");

    Net net;
    try {
        //! [Read and initialize network]
        net = dnn::readNetFromCaffe(modelTxt, modelBin);
        //! [Read and initialize network]
    }
    catch (const cv::Exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        //! [Check that network was read successfully]
        if (net.empty())
        {
            std::cerr << "Can't load network by using the following files: " << std::endl;
            std::cerr << "prototxt:   " << modelTxt << std::endl;
            std::cerr << "caffemodel: " << modelBin << std::endl;
            std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
            std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
            exit(-1);
        }
        //! [Check that network was read successfully]
    }

    if (parser.get<bool>("opencl"))
    {
        net.setPreferableTarget(DNN_TARGET_OPENCL);
    }

    //! [Prepare blob]
    Mat img = imread(imageFile, cv::IMREAD_GRAYSCALE);
    Rect rect(0, 12, img.cols-14, img.rows-12);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }
//    img = img(rect);

    Mat inputBlob = blobFromImage(img, 1.0f, Size(480, 640));   //Convert Mat to batch of images
    inputBlob = (inputBlob - MEAN) / STD;
    std::cout << inputBlob.size[0] << ", " << inputBlob.size[1] << ", " << 
        inputBlob.size[2] << ", " << inputBlob.size[3] << endl;

    //! [Prepare blob]
    net.setInput(inputBlob, "data");        //set the network input
    std::cout << "running model" << endl;
//    Mat result(cv::Size(640, 480), CV_32FC4);
    Mat result =  net.forward();         //compute output
    std::cout << result.size[0] << ", " << result.size[1] << ", " << 
        result.size[2] << ", " << result.size[3] << endl;

    cv::TickMeter t;
    for (int i = 0; i < 10; i++)
    {
        std::cout << i << endl;
        net.setInput(inputBlob, "data");  
        t.start();
        result = net.forward();
        t.stop();
    }

    std::cout <<  "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;


    result = result.reshape(1, 640);
    std::cout << "finished" << endl;

    cv::imshow("test", result);
    cv::waitKey();

    return 0;
} //main
