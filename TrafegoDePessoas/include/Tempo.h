#ifndef TEMPO_H
#define TEMPO_H

#include <ctime>
#include <chrono>
#include <iomanip>


class Tempo
{
    public:
        Tempo();
        void MostrarDiferenca(); //Mostra na tela a diferença
        void SegParaMin(); //Transforma diferença de segundos para minutos
    private:
    //Difereça do tempo atual para o final com difftime
        time_t tempo_inicial;
        time_t tempo_final;
        double minutos;
        double segundos;
};

#endif // TEMPO_H
