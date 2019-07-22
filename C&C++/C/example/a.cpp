# include"stdio.h"
int main()
{
	
	//int score[2][4] = {{2},{2,4}};
	int num[] = {1,2,3};
	printf("num,%d", *(num+2));
	int a = 3;
	int b = 3;
	printf("%d",a++);
	printf("%d",++b);
/*
	for(int i=0;i<2;++i)
		for(int j=0;j<4;++j)
			printf("%d ",score[i][j]);
*/
	return 0;
}
