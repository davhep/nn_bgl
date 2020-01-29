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

using namespace std;

class TrainingData
{
public:
	TrainingData(const string filename);
	bool isEof(void)
	{
		return m_trainingDataFile.eof();
	}
	void getTopology(vector<unsigned> &topology);
	
	// Returns the number of input values read from the file:
	unsigned getNextInputs(vector<double> &inputVals);
	unsigned getTargetOutputs(vector<double> &targetOutputVals);
    void reset(void); 
private:
	ifstream m_trainingDataFile;
};
