#include "pch.h"
#include <iostream>
#include <string>
#include <string.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
	string image_address;
	string save_add;

	Mat image, subimage;
	int index = 1;

	for (int j = 2001; j < 2501; j++)
	{
		image_address = "C:/Users/Ok Subin/Desktop/non-face/";
		image_address += to_string(j);
		image_address += ".jpg";
		image = imread(image_address);

		save_add = "C:/Users/Ok Subin/Desktop/non-face/";
		save_add += to_string(index);
		save_add += ".jpg";

		cout << save_add << endl;

		imwrite(save_add, image);
		index++;		
	}

	return 0;
}