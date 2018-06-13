#include <iostream>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <unistd.h>

#include "DetectorDeFace.h"
#include "Tempo.h"
#include "Estatisticas.h"

int main()
{
	VideoCapture camera(0);
	Mat frame;
	if (!camera.isOpened()) {
		fprintf(stderr, "A camera nao esta aberta!\n"); //Programa fecha se a câmera não abrir
		exit(1);
	}

	face_cascade.load("haarcascade_frontalface_alt.xml"); // Carregar faces

        while(cap.read(frame)) {
		detectFaces(frame); // Função de detecção de faces
		if( waitKey(30) >= 0)    // pause
		break;
	}
  return 0;
}
