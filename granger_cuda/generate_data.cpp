// g++ generate_data.cpp -o generate_data


#include <ctime>
#include <iostream>
#include <fstream>
using namespace std;

int main (int argc, char *argv[]) {
  const int n = atoi(argv[1]);
  srand(time(NULL));

  ofstream x1 ("x1.txt");
  if (x1.is_open())
  {
    for(int count = 0; count < n; count ++){
        x1 << rand()/(RAND_MAX/10.0)<< "\n" ;
    }
    x1.close();
  }
  else cout << "Unable to open file";

    ofstream x2 ("x2.txt");
  if (x2.is_open())
  {
    for(int count = 0; count < n; count ++){
        x2 << rand()/(RAND_MAX/10.0)<< "\n" ;
    }
    x2.close();
  }
  else cout << "Unable to open file";
  return 0;
}