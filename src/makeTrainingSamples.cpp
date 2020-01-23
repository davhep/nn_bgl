#include <iostream>
#include <cmath>
#include <cstdlib>

using namespace std;

int main()
{
	// Random training sets two inputs and one output

	cout << "topology: 2 4 1" << endl;
	for(int i = 2000; i >= 0; --i)
	{
		double n1 = (2.0 * rand() / double(RAND_MAX));
		double n2 = (2.0 * rand() / double(RAND_MAX));
		//double t = sqrt(n1*n1+n2*n2)/10;
		double t = 0;
		if( (n1*n1+n2*n2 <4) && (n1*n1+n2*n2 >2 )) t = 0.5; // ring of value 0.5
		cout << "in: " << n1 << " " << n2 << " " << endl;
		cout << "out: " << t << " " << endl; 
	}

}
