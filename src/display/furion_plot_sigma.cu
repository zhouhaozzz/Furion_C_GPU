#include "Furion_Plot_Sigma.h"

using namespace Furion_NS;

Furion_Plot_Sigma::Furion_Plot_Sigma()
{

}

Furion_Plot_Sigma::~Furion_Plot_Sigma()
{

}

void Furion_Plot_Sigma::Furion_plot_sigma(real_t* X, real_t* Y, real_t* Phi, real_t* Psi, int rank1)
{
	ofstream fileout("data/Furion_plot_sigma_" + inttoStr(rank1 + 1) + ".dat");
	fileout << std::fixed;
	fileout << std::setprecision(15);

	for (int i = 0; i < Furion::n; i++)
	{
		fileout << X[i] << " ";
	}
	fileout << "\n";
	for (int i = 0; i < Furion::n; i++)
	{
		fileout << Y[i] << " ";
	}
	fileout << "\n";
	for (int i = 0; i < Furion::n; i++)
	{
		fileout << Phi[i] << " ";
	}
	fileout << "\n";
	for (int i = 0; i < Furion::n; i++)
	{
		fileout << Psi[i] << " ";
	}
	fileout << "\n";

	fileout.close();
}

string Furion_Plot_Sigma::inttoStr(int s)
{
	string c = "";
	int m = s;
	int n = 0;

	while (m > 0)
	{
		n++;
		m /= 10;
	}

	for (int i = 0; i < n; i++)
	{
		c = (char)(s % 10 + 48) + c;
		s /= 10;
	}
	return c;
}