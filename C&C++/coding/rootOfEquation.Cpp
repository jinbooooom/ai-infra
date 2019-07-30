/*
已知f(x)=3x^3-2x^2+5x-16在（-2，4）上单调且有零点，
用二分法求方程f(x)=0在（-2，4）之间的根，要求精度为10^-6。
*/

#include <iostream>
#include <cmath>
using namespace std;

bool differentSign(double x1,double x2)
{
	return x1 * x2 <= 0;
}

double f(double x)//设置函数f(x)
{
	return 3 * pow(x, 3) - 2 * pow(x, 2) + 5 * x - 16;
}

int main(int argc, char *argv[])
{
	double x1 = -2.0, x2 = 4.0;
	double x = (x1 + x2)/2.0;
	while(fabs(f(x)) > 1e-6)
	{
		if(differentSign(f(x1), f(x)))
			x2 = x;
		else
			x1 = x;	
		x = (x1 + x2)/2.0;
	}
	printf("%lf\n",x);
	cout << x << endl;//printf的输出比cout更详细一些，但cout更灵活。
	return 0;
}
