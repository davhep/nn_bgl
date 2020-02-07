#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

void fill_data(std::string file_name){
	ofstream file(file_name);
	for(int i = 2000; i >= 0; --i)
	{
		double n1 = (2.0 * rand() / double(RAND_MAX));
		double n2 = (2.0 * rand() / double(RAND_MAX));
		//double t = sqrt(n1*n1+n2*n2)/10;
		double t = 0;
		if( (n1*n1+n2*n2 <4) && (n1*n1+n2*n2 >2 )) t = 0.5; // ring of value 0.5
		file << "in: " << n1 << " " << n2 << " " << endl;
		file << "out: " << t << " " << endl; 
	}
	file.close();
}

int main()
{
	// Random training sets two inputs and one output
	ofstream topology("topology.txt");
	topology << "topology: 2 4 1" << endl;
	topology.close();
	
	fill_data("train_data.txt");
	fill_data("validate_data.txt");
}
