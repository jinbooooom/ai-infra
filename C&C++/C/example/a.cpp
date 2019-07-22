# include"stdio.h"
int main()
{
	
	int score[2][4] = {{2},{2,4}};
	for(int i=0;i<2;++i)
		for(int j=0;j<4;++j)
			printf("%d ",score[i][j]);
	return 0;
}
