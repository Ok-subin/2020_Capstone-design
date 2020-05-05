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
	string inform_address = "C:/Users/Ok Subin/Desktop/FDDB-folds/FDDB-fold-01-ellipseList.txt";   //	FDDB - fold - 01 - ellipseList
	string image_address = "C:/Users\/k Subin/Desktop/originalPics/";

	int total = 0;
	ifstream file(inform_address);
	int** inform = new int*[total];;
	Mat image;

	string line;
	char ch_line[200];
	int index = 0;
	int arr_idx = 0, blank_idx1 = 0, blank_idx2 = 0, num_idx = 0, blank_idx = 0;

	while (getline(file, line))
	{
		// 시작줄
		if (line.find("2002/") != string::npos || line.find("2003/") != string::npos)
		{
			index = 0;
			arr_idx = 0;
			num_idx = 0;
			image_address += line;
			image_address += ".jpg";

			// 이미지 불러오기
			image = imread(image_address);
			cout << image_address << "\n";

			image_address = "C:/Users\/k Subin/Desktop/originalPics/";

			index++;

		}

		else
		{
			if (index == 1)
			{
				// total : 이미지 속 총 얼굴의 수
				total = stoi(line);
				cout << "total : " << total << "\n";

				for (int i = 0; i < total; i++)
				{
					inform[i] = new int[5];
				}

				index++;
			}

			else
			{
				// arr_idx는 total 개수 중 몇번째까지 했는지, inform 배열에 넣기 위해 필요				
				// split
				string num_s;
				float num = 0.;

				while (true)
				{
					cout << "Start" << "\n";
					if (blank_idx > 5)
					{
						blank_idx = 0;
						blank_idx1 = 0;
						blank_idx2 = 0;

						arr_idx++;

						if (arr_idx == total - 1)
						{
							break;
						}
					}

					else
					{
						if (num_idx == 0)
						{
							blank_idx1 = 0;
							blank_idx2 = line.find(" ");

							blank_idx++;
						}

						else
						{
							blank_idx1 = blank_idx2;
							blank_idx2 = line.find(" ", blank_idx1 + 1);
							blank_idx++;
						}

						cout << "bl_01 : " << blank_idx1 << ", bl_02 : " << blank_idx2 << "\n";
						num_s = line.substr(blank_idx1, blank_idx2 - blank_idx1);
						num = stof(num_s);
						cout << "num : " << num << "\n";

						//<major_axis_radius minor_axis_radius angle center_x center_y 1>.
						// inform도 여러 이미지에 대해 처리하는 부분 위해서 수정해야 함
						inform[arr_idx][num_idx] = num;
						num_idx++;
						index++;
						arr_idx++;

					}

				}
				float left, right, top, bottom;
				float rt1_left, rt1_right, rt1_top, rt1_bottom;
				float rt2_left, rt2_right, rt2_top, rt2_bottom = 0;

				for (int a = 0; a < total; a++)
				{
					rt1_left = inform[a][3] - inform[a][3] * cos(inform[a][2]);
					rt1_right = inform[a][3] + inform[a][3] * cos(inform[a][2]);
					rt1_top = inform[a][4] - inform[a][3] * sin(inform[a][2]);
					rt1_bottom = inform[a][4] + inform[a][3] * sin(inform[a][2]);


					rt2_left = inform[a][3] - inform[a][1] * cos(inform[a][2] - 3.141592/2);
					rt2_right = inform[a][3] + inform[a][1] * cos(inform[a][2] - 3.141592/2);
					rt2_top = inform[a][4] - inform[a][1] * sin(inform[a][2]+3.141592/2);
					rt2_top = inform[a][4] + inform[a][1] * sin(inform[a][2]+3.141592/2);

					left = rt1_left < rt2_left ? rt1_left : rt2_left;
					right = rt1_right > rt2_right ? rt1_right : rt2_right;
					top = rt1_top > rt2_top ? rt1_top : rt2_top;
					bottom = rt1_bottom < rt2_bottom ? rt1_bottom : rt2_bottom;
				}

				cout << (left + right) / 2 << " " << (top + bottom) / 2 << " " << (left + right) << " " << (top + bottom) << endl;
				Rect rect((left + right) / 2, (top + bottom) / 2, (left + right), (top + bottom));

				Mat subimage = image(rect);
				//imshow("image", subimage);
				//waitKey(0);

				string image_name = "C:/Users/Ok Subin/Desktop/image";
				int image_num = 0;

				image_name += image_num;
				image_name += ".jpg";
				
				imwrite("image.jpg", image);
				imwrite("image.jpg", subimage);

				image_num++;


			}
		}

	}

	return 0;
}