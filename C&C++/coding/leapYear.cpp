/*
闰年是公历中的名词。闰年分为普通闰年和世纪闰年。
普通闰年；公历年份是4的倍数的，一般是闰年。（如2004年就是闰年）；
世纪闰年:公历年份是整百数的，必须是400的倍数才是闰年（如1900年不是世纪闰年，2000年是世纪闰年）；
其他年份为平年。
*/
#include<iostream>

using namespace std;

bool leapYear(int year)
{
	return (year % 100) && (year % 4 == 0) || (year % 100 == 0) && (year % 400 == 0);
}

int main()
{
	int year;
	cin >> year;
	cout << (leapYear(year) ? "闰年" : "平年") << endl;
}

