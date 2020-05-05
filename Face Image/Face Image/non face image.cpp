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
	string save_add, save_add1, save_add2, save_add3, save_add4;

	Mat image, image2, image3, image4;
	Mat subimage1, subimage2, subimage3, subimage4;
	int index = 673;
	
	for (int j = 1; j < 673; j++)
	{
		image_address = "C:/Users/Ok Subin/Desktop/non_face/";
		image_address += to_string(j);
		image_address += ".jpg";

		cout << image_address << endl;

		image = imread(image_address);

		resize(image, image, Size(600, 600));

		for (int i = 0; i < 4; i++)
		{
			save_add = "C:/Users/Ok Subin/Desktop/non_face/";
			cout << i << endl;
			/*
			switch (i)
			{
			case 0:
				subimage = image(Range(0, 300), Range(0, 300));
				break;
			case 1:
				subimage = image(Range(300, 600), Range(0, 300));
				break;
			case 2:
				subimage = image(Range(0, 300), Range(300, 600));
				break;
			case 3:
				subimage = image(Range(300, 600), Range(300, 600));
				break;
			default:
				break;
			} */

			switch (i)
			{
			case 0:
				subimage1 = image(Range(0, 300), Range(0, 300));
				subimage2 = image(Range(0, 300), Range(100, 400));
				subimage3 = image(Range(0, 300), Range(200, 500));
				subimage4 = image(Range(0, 300), Range(300, 600));
				break;

			case 1:
				subimage1 = image(Range(100, 400), Range(0, 300));
				subimage2 = image(Range(100, 400), Range(100, 400));
				subimage3 = image(Range(100, 400), Range(200, 500));
				subimage4 = image(Range(100, 400), Range(300, 600));

				break;

			case 2:
				subimage1 = image(Range(200, 500), Range(0, 300));
				subimage2 = image(Range(200, 500), Range(100, 400));
				subimage3 = image(Range(200, 500), Range(200, 500));
				subimage4 = image(Range(200, 500), Range(300, 600));
				break;

			case 3:
				subimage1 = image(Range(300, 600), Range(0, 300));
				subimage2 = image(Range(300, 600), Range(100, 400));
				subimage3 = image(Range(300, 600), Range(200, 500));
				subimage4 = image(Range(300, 600), Range(300, 600));
				break;

			default:
				break;
			}

			save_add1 = save_add + to_string(index) + ".jpg";
			save_add2 = save_add + to_string(index+1) + ".jpg";
			save_add3 = save_add + to_string(index+2) + ".jpg";
			save_add4 = save_add + to_string(index+3) + ".jpg";

			cout << save_add << endl;

			imwrite(save_add1, subimage1);
			imwrite(save_add2, subimage2);
			imwrite(save_add3, subimage3);
			imwrite(save_add4, subimage4);

			index += 4;
		}
	}
	return 0;
}