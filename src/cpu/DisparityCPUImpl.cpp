#include "DisparityCPUImpl.h"

int getIndexGlobal(std::size_t countX, int i, int j)
{
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j)
{
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

// implementation on CPU
void DisparityMappingCPUImpl(const std::vector<float>& leftInputImage, const std::vector<float>& rightInputImage, std::vector<float>& output, size_t countX, size_t countY, int isSSD)
{
	int windowSize = 17;
	int windowRange = windowSize / 2;

	float value = 0.0;
	float min = 8000.0;

	float totaldisparity = 0.0;
	int disparityMax = 70;
	

	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			min = 10000.0;
			for (int disparity = 0; disparity <= disparityMax; disparity++)
			{
				value = 0;
				for (int width = 0; width < windowRange *2; width++)
				{
					for (int height = 0; height < windowRange *2; height++)
					{
						float disp = 0;
						disp = abs(getValueGlobal(leftInputImage, countX, countY, width + i, height + j) - getValueGlobal(rightInputImage, countX, countY, width + i - disparity, height + j));
						if (isSSD == 1)
						{
							value = value + (disp * disp);
						}
						else
						{
							value = value + disp;
						}
					}
				}
				if (min > value)
				{
					min = value;
					totaldisparity = (float)(disparity) / disparityMax;
				}
			}
			output[getIndexGlobal(countX, i, j)] = totaldisparity;
		}

	}
}