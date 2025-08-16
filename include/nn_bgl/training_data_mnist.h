#include <nn_bgl/training_data.h>

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <map>  
#include <numeric>

#include <iostream>
#include <fstream>

#include <iterator>
#include <stdint.h>

class TrainingDataMnist : public TrainingData {
public:
	bool isEof(void);
	void getNextInputs(vector<double> &inputVals);
	void getTargetOutputs(vector<double> &targetOutputVals);
	void reset(void); 
	void InitFile(const string filename_images, const string filename_labels);
	unsigned int index_max=60000, index=0;
private:
	std::ifstream images;
	std::ifstream labels;
};
