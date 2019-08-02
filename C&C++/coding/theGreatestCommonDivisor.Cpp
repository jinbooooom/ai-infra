/*
求两个正整数的最大公约数
1、辗转相除法f1()：
2、更相减损法f2()：
*/

#include<iostream>
using namespace std;

int f1(int a, int b)
{
	int t = 0;
	if(a < b)
	{
		t = a;
		a = b;
		b = t;
	}
	if(t = a%b) //a除以b的余数为零，说明b是最大公约数，直接返回b
	{
		printf("%d / %d 余 %d\n",a,b,t);
		a = a>b?a:b;
		f1(b, t);
	}
	else	
	{
		printf("%d / %d 整除\n",a,b);
		return b;
	}
}

int f2(int a, int b)
{
	int t = 0;
	if(a < b)
	{
		t = a;
		a = b;
		b = t;
	}
	if(t = a - b) //当差为零，说明被减数与减数相等，此时减数就是最大公约数
	{
		printf("%d - %d = %d\n", a, b, t);
		f2(b, t);
	}
	else
	{
		printf("%d - %d = %d\n", a, b, t);
		return a;
	}
}


int main()
{
	int a, b;
	cin >> a >> b;
	cout << "辗转相除法: " << f1(a, b) << endl << endl;
	cout << "更相减损法: " << f2(a, b) << endl;
}