#include "helper.hpp"

using namespace std;

// Return time in ms
double timestamp() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return((double) (tp.tv_sec*1000.0 + tp.tv_usec/1000.0));
}
