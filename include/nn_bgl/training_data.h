#include <vector>
#include <iostream>

using namespace std;

class TrainingData{
public:
	virtual bool isEof(void)=0;
	virtual void getNextInputs(vector<double> &inputVals)=0;
	virtual void getTargetOutputs(vector<double> &targetOutputVals)=0;
	void get(vector<double> &inputVals, vector<double> &targetOutputVals){
		getNextInputs(inputVals);
		getTargetOutputs(targetOutputVals);
	};
    virtual void reset(void)=0; 
};
