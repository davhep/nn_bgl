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

class TrainingDataHuman : public TrainingData {
public:
	bool isEof(void);
	void getNextInputs(vector<double> &inputVals);
	void getTargetOutputs(vector<double> &targetOutputVals);
	void reset(void); 
	void InitFile(const string filename);
    void ReadAllFromFile(vector<std::pair<vector<double>, vector<double>>> &input_output_vals, int input_size, int output_size);
private:
	ifstream m_trainingDataFile;
};
