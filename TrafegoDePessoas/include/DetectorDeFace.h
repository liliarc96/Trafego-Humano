#ifndef DETECTORDEFACE_H
#define DETECTORDEFACE_H
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

class DetectorDeFace
{
    public:
        void detectFaces(Mat frame);
    private:
};

#endif // DETECTORDEFACE_H
