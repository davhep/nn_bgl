#include<./training_data.h>

using namespace std;

void TrainingData::reset(void)
{
	m_trainingDataFile.clear();
	m_trainingDataFile.seekg (0);
}

void TrainingData::getTopology(std::string topology_file_name, vector<unsigned> &topology)
{
	string line;
	string label;
	
	ifstream topology_file(topology_file_name);
	getline(topology_file, line);
	topology_file.close();
	
	stringstream ss(line);
	ss >> label;
	if(this->isEof() || label.compare("topology:") != 0)
	{
		abort();
	}

	while(!ss.eof())
	{
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}
	return;
}

TrainingData::TrainingData(const string filename)
{
	m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
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
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
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

    return targetOutputVals.size();
}

unsigned TrainingData::get(vector<double> &inputVals, vector<double> &targetOutputVals)
{
	getNextInputs(inputVals);
	getTargetOutputs(targetOutputVals);
}
