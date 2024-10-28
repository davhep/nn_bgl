#include<./training_data_human.h>

using namespace std;

bool TrainingDataHuman::isEof(void)
{
		return m_trainingDataFile.eof();
};
	
void TrainingDataHuman::reset(void)
{
	m_trainingDataFile.clear();
	m_trainingDataFile.seekg (0);
}

void TrainingDataHuman::InitFile(const string filename)
{
	m_trainingDataFile.open(filename.c_str(), ios::in);
	assert(m_trainingDataFile.is_open());
}


void TrainingDataHuman::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);
	string label;
    ss >> label;

    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            //std::cout << oneValue << "  ";
            inputVals.push_back(oneValue);
        }
        //std::cout  << std::endl;
    }
}

void TrainingDataHuman::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

	string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }
}

void TrainingDataHuman::ReadAllFromFile(vector<std::pair<vector<double>, vector<double>>> &input_output_vals, int input_size, int output_size){
    vector<double> inputs, outputs;
    while(!m_trainingDataFile.eof()){
        getNextInputs(inputs);
        getTargetOutputs(outputs);
        if(inputs.size() == input_size && outputs.size() == output_size)
            input_output_vals.push_back(std::pair(inputs, outputs));
        else{
            cout << "FAIL in input/output sizes!!!!" << endl;
        }
    }
}
