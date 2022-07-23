
// includes
#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>
#include "DisparityCPUImpl.h"
#include "OpenCLConfig.h"

//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	OpenCLConfig config;
	if (!config.createPlatform()) {
		std::cout << "Platform ceation failed" << std::endl;
		return 0;
	}

	config.createContext();

	KernelParameter param;
	param.wgSizeX = 16;
	param.wgSizeY = 16;
	param.countX = param.wgSizeX * 24;
	param.countY = param.wgSizeY * 18;

	param.count = param.countX * param.countY;
	param.size = param.count * sizeof(float);

	config.setKernelParameter(param);

	//Allocate space for input and output data from CPU and GPU.
	std::vector<float> inputLeftImage(config.getKernelParameter().count);
	std::vector<float> inputRightImage(config.getKernelParameter().count);
	std::vector<float> outputGpuSSD(config.getKernelParameter().count);
	std::vector<float> outputCpuSSD(config.getKernelParameter().count);

	int deviceNumber = argc < 2 ? 1 : atoi(argv[1]);
	config.createCommandQueue(deviceNumber);

	memset(inputLeftImage.data(), 255, config.getKernelParameter().size);
	memset(inputRightImage.data(), 255, config.getKernelParameter().size);
	memset(outputGpuSSD.data(), 255, config.getKernelParameter().size);
	memset(outputCpuSSD.data(), 255, config.getKernelParameter().size);

	//	Read input images and set the data in respective buffers
	std::vector<float> readLeftImage;
	std::vector<float> readRightImage;

	std::size_t leftImageWidth, leftImageHeight, rightImageWidth, rightImageHeight;

	Core::readImagePGM("images/Decoration_Left.pgm", readLeftImage, leftImageWidth, leftImageHeight);
	Core::readImagePGM("images/Decoration_Right.pgm", readRightImage, rightImageWidth, rightImageHeight);

	for (size_t j = 0; j < config.getKernelParameter().countY; j++)
	{
		for (size_t i = 0; i < config.getKernelParameter().countX; i++)
		{
			inputLeftImage[i + config.getKernelParameter().countX * j] = readLeftImage[(i % leftImageWidth) + leftImageWidth * (j % leftImageHeight)];
			inputRightImage[i + config.getKernelParameter().countX * j] = readRightImage[(i % rightImageWidth) + rightImageWidth * (j % rightImageHeight)];
		}
	}

	// when i = 0 -> SAD : i = 1 -> SSD
	for (int i = 0; i < 2; i++)
	{
		//CPU Implementation invocation
		std::cout << "-----------	CPU Execution Started" << ((i == 0) ? " -> SAD " : " -> SSD ") << "	----------- \n" << std::endl;
		Core::TimeSpan cpuStartTime = Core::getCurrentTime();

		DisparityMappingCPUImpl(inputLeftImage, inputRightImage, outputCpuSSD, config.getKernelParameter().countX, config.getKernelParameter().countY, i);
		Core::TimeSpan cpuEndTime = Core::getCurrentTime();

		Core::TimeSpan cpuExecutionTime = cpuEndTime - cpuStartTime;
		std::cout << "		CPU Execution Time : " << ((i == 0) ? " -> SAD " : " -> SSD ") << cpuExecutionTime << "\n" << std::endl;
		std::cout << "-----------	CPU Execution End" << ((i == 0) ? " -> SAD " : " -> SSD ") << "	----------- \n" << std::endl;

		//Store output Image -- CPU
		std::string name = "disparitymap_" + std::string((i==0 ? "SAD" : "SSD")) + "_cpu.pgm";
		Core::writeImagePGM(name, outputCpuSSD, config.getKernelParameter().countX, config.getKernelParameter().countY);


		//GPU Implementation invocation
		std::cout << "-----------	GPU Execution Start" << ((i == 0) ? " -> SAD " : " -> SSD ") << "	----------- \n" << std::endl;

		config.setCLFilePath("DisparityMap.cl");
		Core::TimeSpan gpuExecutionTime = config.executeKernel(outputCpuSSD, inputLeftImage, inputRightImage, i);

		std::cout << "-----------	GPU Execution End" << ((i == 0) ? " -> SAD " : " -> SSD ") << "	----------- " << "\n"  << std::endl;

		std::cout << "************************************************************"<<std::endl;
		std::cout << "	GPU Speedup over CPU with method" << ((i == 0) ? " -> SAD " : " -> SSD ")  << cpuExecutionTime.getSeconds() / gpuExecutionTime.getSeconds() << std::endl;
		std::cout << "************************************************************\n"<< std::endl;

	}

}
