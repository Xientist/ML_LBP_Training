#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <fstream>

using namespace cv;

const int CORRECTLY_MASKED = 1;
const int INCORRECTLY_MASKED = 0;

// Structure pour sélectionner chaque voisin d'un pixel de l'image pour le traitement
struct Voisinage {

	uchar voisins[8];
	uchar centre;
};

void creerVoisinage(const Mat &m, int x, int y, Voisinage &v) {

	v.centre = m.at<uchar>(x, y);

	v.voisins[0] = m.at<uchar>(x-1, y-1);
	v.voisins[1] = m.at<uchar>(x, y - 1);
	v.voisins[2] = m.at<uchar>(x + 1, y - 1);
	v.voisins[3] = m.at<uchar>(x - 1, y);
	v.voisins[4] = m.at<uchar>(x + 1, y);
	v.voisins[5] = m.at<uchar>(x - 1, y + 1);
	v.voisins[6] = m.at<uchar>(x, y + 1);
	v.voisins[7] = m.at<uchar>(x + 1, y + 1);
}

void creerVoisinageCouleur(const Mat &m, int channel, int x, int y, Voisinage &v) {

	v.centre = m.at<Vec3b>(x, y)[channel];

	v.voisins[0] = m.at<Vec3b>(x - 1, y - 1)[channel];
	v.voisins[1] = m.at<Vec3b>(x, y - 1)[channel];
	v.voisins[2] = m.at<Vec3b>(x + 1, y - 1)[channel];
	v.voisins[3] = m.at<Vec3b>(x - 1, y)[channel];
	v.voisins[4] = m.at<Vec3b>(x + 1, y)[channel];
	v.voisins[5] = m.at<Vec3b>(x - 1, y + 1)[channel];
	v.voisins[6] = m.at<Vec3b>(x, y + 1)[channel];
	v.voisins[7] = m.at<Vec3b>(x + 1, y + 1)[channel];
}

// retourne l'histogramme (tableau d'entier à 256 cases) du filtre LBP de l'image (cv::Mat) 'image'
void LBP(Mat image, int* histogramme) {

	for (int x = 1; x < image.rows-1; x++) {
		
		for (int y = 1; y < image.cols-1; y++) {

			Voisinage v;
			creerVoisinage(image, x, y, v);
			int octet = 0;

			for (int i = 0; i < 8; i++) {

				octet = octet << 1;
				octet += (v.voisins[i] > v.centre) ? 1 : 0;
			}

			histogramme[octet]++;
		}
	}
}

// filtre LBP appliqué à chaque couche RGB de l'image
void LBPColor(Mat image, int* histogramme) {

	for (int channel = 0; channel < 3; channel++) {

		for (int x = 1; x < image.rows - 1; x++) {

			for (int y = 1; y < image.cols - 1; y++) {

				Voisinage v;
				creerVoisinageCouleur(image, channel, x, y, v);
				int octet = 0;

				for (int i = 0; i < 8; i++) {

					octet = octet << 1;
					octet += (v.voisins[i] > v.centre) ? 1 : 0;
				}

				histogramme[octet]++;
			}
		}
	}
}

int main()
{

	int nbImages = 1000;
	int offset = nbImages / 2;
	int numDataset = 1;

	std::ofstream training("training.txt");

	for (int i = 0; i < nbImages; i++) {

		Mat image;
		if (i < offset) {
			image = imread("C:/Users/Mad Scientifique/Documents/m2_numerisation/dataset_masks/COLOR/" + std::to_string(numDataset) + "TRAIN/CMFD/image_" + std::to_string(i) + ".jpg", IMREAD_COLOR);
		}
		else {
			image = imread("C:/Users/Mad Scientifique/Documents/m2_numerisation/dataset_masks/COLOR/" + std::to_string(numDataset) + "TRAIN/IMFD/image_" + std::to_string(i-offset) + ".jpg", IMREAD_COLOR);
		}
		
		if (image.empty())
		{
			std::cout << "L'image n'a pas pu être lue." << std::endl;
			return 1;
		}

		int hist[256] = { 0 };

		LBPColor(image, hist);

		for (int i = 0; i < 256; i++) {

			training << hist[i] << " ";
		}

		training << ((i < offset) ? CORRECTLY_MASKED : INCORRECTLY_MASKED) << std::endl;

		image.release();
	}
	
	training.close();

	std::ofstream test("test.txt");

	for (int i = 0; i < nbImages; i++) {

		Mat image;
		if (i < offset) {
			image = imread("C:/Users/Mad Scientifique/Documents/m2_numerisation/dataset_masks/COLOR/" + std::to_string(numDataset) + "TEST/CMFD/image_" + std::to_string(i) + ".jpg", IMREAD_COLOR);
		}
		else {
			image = imread("C:/Users/Mad Scientifique/Documents/m2_numerisation/dataset_masks/COLOR/" + std::to_string(numDataset) + "TEST/IMFD/image_" + std::to_string(i-offset) + ".jpg", IMREAD_COLOR);
		}

		if (image.empty())
		{
			std::cout << "L'image n'a pas pu être lue." << std::endl;
			return 1;
		}

		int hist[256] = { 0 };

		LBPColor(image, hist);


		for (int i = 0; i < 256; i++) {

			test << hist[i] << " ";
		}

		test << ((i < offset) ? CORRECTLY_MASKED : INCORRECTLY_MASKED) << std::endl;

		image.release();
	}

	test.close();
}