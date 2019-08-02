/*
将一个正整数分解成质因子的连乘积
*/

#include<iostream>
#include<vector>

using namespace std;

void print(vector<int> vector1)
{
	for (int i = 0; i < vector1.size() - 1; i++)
		cout << vector1[i] << " * ";
	cout << vector1[vector1.size() - 1] << endl;
}

vector<int> primeFactor(int number)
{
	vector<int> prime;
	for (int i = 2; i <= number / 2; ++i)
		/* i <= number/2，质因数 i 只会出现在 number 的前半部分。
		相比于 i <= number 也能减少一些计算量。*/
	{
		if (number % i == 0)
		{
			number /= i;
			prime.push_back(i);
			i = 2;
			/* 找到了一个质因数 i 后，number = number/i，
			并且 下一个质因数 i 又得重新从2开始找。*/
		}
	}
	prime.push_back(number);//因为采用 i <= number/2 的原因，在该范围内没有找到质因子，说明 number 本身就是质数。
	return prime;
}

int main()
{
	int number;
	cin >> number;
	cout << number << "的质因子连乘积为：";
	print(primeFactor(number));
}
