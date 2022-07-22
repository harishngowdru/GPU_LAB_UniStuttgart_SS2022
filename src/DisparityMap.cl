#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) 
{
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float getValueImage(__read_only image2d_t a, int i, int j)
{
	return read_imagef(a, sampler, (int2) { i, j }).x;
}


// Calculate square of the difference of two input pixels and return the result (used for SSD Calculation)

float calculateDifference(float value1, float value2, int isSSD)
{
	float difference;
	difference = value1 - value2;
	// when i = 0 -> SAD : i = 1 -> SSD
	if (isSSD == 1)
	{
		return difference * difference;
	}
	else
	{
		if (difference < 0)
			return (-1 * difference);
		else
			return difference;
	}
	
}

// Kernel for Disparity mapping using SAD and SSD algorithm

__kernel void disparityMap(__read_only image2d_t leftImage, __read_only image2d_t rightImage, __global float* output, int isSSD)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int windowSize = 13;
	int windowRange = windowSize / 2;
	float value = 0.0;
	float min = 10000.0;
	float disparityintotal = 0.0;
	int disparityMax = 100;
	float temp;

	for (int disparity = 0; disparity <= disparityMax; disparity++)
	{
		value = 0;
		for (int imageWidth = 0; imageWidth < windowRange*2; imageWidth++)
		{
			for (int imageHeight = 0; imageHeight < windowRange *2; imageHeight++)
			{

				value = value + calculateDifference(getValueImage(leftImage, imageWidth + i, imageHeight + j), getValueImage(rightImage, imageWidth + i - disparity, imageHeight + j), isSSD);
			}
		}

		if (value < min)
		{
			min = value;
			disparityintotal = (float)(disparity) / disparityMax;
		}

	}

	output[getIndexGlobal(countX, i, j)] = disparityintotal;
}
