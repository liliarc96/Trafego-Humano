#include <iostream>
#include "Tempo.h"
using namespace std;
Tempo::Tempo()
{
// Get current time
std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
tempo_inicial = time_now;
//Quando a camera for fechada:
std::chrono::time_point<std::chrono::system_clock> time_now = std::chrono::system_clock::now();
tempo_final = time_now;
intervalo = difftime (tempo_final, tempo = inicial); // compara o tempo

// Print time using ctime
//std::time_t time_now_t = std::chrono::system_clock::to_time_t(time_now);
//std::cout << std::ctime(&time_now_t) << std::endl;

//OU
/*int main ()
{
  time_t now;
  struct tm newyear;
  double seconds;

  time(&now);  /* get current time; same as: now = time(NULL)  */

  newyear = *localtime(&now);

  newyear.tm_hour = 0; newyear.tm_min = 0; newyear.tm_sec = 0;
  newyear.tm_mon = 0;  newyear.tm_mday = 1;

  seconds = difftime(now,mktime(&newyear));

  printf ("%.f seconds since new year in the current timezone.\n", seconds);

return 0;
}*/
}
