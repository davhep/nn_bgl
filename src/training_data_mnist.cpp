#include<./training_data_mnist.h>

#ifndef IMAGE_SIZE
#define IMAGE_SIZE 784
#endif

using namespace std;

bool TrainingDataMnist::isEof(void)
{
	return index < index_max;
};

void TrainingDataMnist::reset(void)
{
	index = 0;
};

void TrainingDataMnist::InitFile(const string filename_images, const string filename_labels)
{
	images.open(filename_images, std::ios::binary | std::ios::in);
	labels.open(filename_labels, std::ios::binary | std::ios::in);

	if (images.fail() || labels.fail())
		exit(1);
}

void TrainingDataMnist::getNextInputs(vector<double> &inputVals)
{
	images.seekg(16 + IMAGE_SIZE*index, std::ios::beg);
    inputVals.clear();
	
	uint8_t buffer [IMAGE_SIZE];
	images.read((char*) buffer, IMAGE_SIZE);

	for (int i = 0; i < IMAGE_SIZE; i++)
	{
		inputVals.push_back((float)buffer[i] / 255.0);
	}
	index++;

}


// What value is expected
void TrainingDataMnist::getTargetOutputs(vector<double> &targetOutputVals)
{
	if(targetOutputVals.size() != 10) targetOutputVals.resize(10);
	labels.seekg(8 + index, std::ios::beg);
	char x;
	labels.read(&x, 1);
	std::fill(targetOutputVals.begin(), targetOutputVals.end(), 0);
	targetOutputVals[x]=1;

}
