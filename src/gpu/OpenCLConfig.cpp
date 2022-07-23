#include "OpenCLConfig.h"

bool OpenCLConfig::createPlatform()
{
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return false;
	}

	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}

	return true;
}

void OpenCLConfig::createContext()
{
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	context = cl::Context(CL_DEVICE_TYPE_GPU, prop);
}

cl::Context OpenCLConfig::getContext()
{
	return context;
}

void OpenCLConfig::setKernelParameter(KernelParameter param)
{
	kernelParameter = param;
}

KernelParameter OpenCLConfig::getKernelParameter()
{
	return kernelParameter;
}

void OpenCLConfig::createCommandQueue(int deviceNumber)
{
	// Get a device of the context
	std::cout << "Using device " << deviceNumber << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNumber > 0);
	ASSERT((size_t)deviceNumber <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNumber - 1];

	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);
	std::cout << std::endl;

	// Create a command queue
	queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
}

void OpenCLConfig::setCLFilePath(std::string path)
{
	clFilePath = path;
}

Core::TimeSpan OpenCLConfig::executeKernel(std::vector<float>& outputGpuSSD, std::vector<float>& inputLeftImage, std::vector<float>& inputRightImage, bool isSSD)
{
	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, clFilePath);
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Creation of Image - inputs
	cl::Image2D imageL(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), kernelParameter.countX, kernelParameter.countY);
	cl::Image2D imageR(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), kernelParameter.countX, kernelParameter.countY);

	// Buffer creations
	cl::Buffer gpu_output(context, CL_MEM_READ_WRITE, (kernelParameter.countX * kernelParameter.countY) * sizeof(int));

	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = kernelParameter.countX;
	region[1] = kernelParameter.countY;
	region[2] = 1;

	// Reinitialize output memory to 0xff
	memset(outputGpuSSD.data(), 255, kernelParameter.size);

	// Copy input data to device
	queue.enqueueWriteBuffer(gpu_output, true, 0, kernelParameter.size, outputGpuSSD.data());

	//Enqueue the images to the kernel
	cl::Event writeEvent;
	queue.enqueueWriteImage(imageL, true, origin, region, kernelParameter.countX * (sizeof(float)), 0, inputLeftImage.data(), NULL, &writeEvent);
	queue.enqueueWriteImage(imageR, true, origin, region, kernelParameter.countX * (sizeof(float)), 0, inputRightImage.data(), NULL, &writeEvent);

	// Create kernel object
	cl::Kernel disparityMapping(program, "disparityMap");

	// Set Kernel Arguments
	cl::Event kernelExecution;
	disparityMapping.setArg<cl::Image2D>(0, imageL);
	disparityMapping.setArg<cl::Image2D>(1, imageR);
	disparityMapping.setArg<cl::Buffer>(2, gpu_output);
	disparityMapping.setArg<cl_int>(3, isSSD);

	//Launch Kernel on the device
	queue.enqueueNDRangeKernel(disparityMapping, 0, cl::NDRange(kernelParameter.countX, kernelParameter.countY), cl::NDRange(kernelParameter.wgSizeX, kernelParameter.wgSizeY), NULL, &kernelExecution);

	// Copy output data from GPU back to host
	cl::Event readEvent;
	queue.enqueueReadBuffer(gpu_output, true, 0, kernelParameter.count * sizeof(int), outputGpuSSD.data(), NULL, &readEvent);

	//Store the output image -- GPU
	std::string name = "disparitymap_" + std::string((isSSD ? "SAD" : "SSD")) + "_gpu.pgm";
	Core::writeImagePGM(name, outputGpuSSD, kernelParameter.countX, kernelParameter.countY);

	// Print performance data
	Core::TimeSpan gpuStartTime = OpenCL::getElapsedTime(kernelExecution);
	Core::TimeSpan readWriteTime = OpenCL::getElapsedTime(writeEvent) + OpenCL::getElapsedTime(readEvent);
	Core::TimeSpan gpuExecutionTime = gpuStartTime + readWriteTime;

	std::cout << "		GPU Read/Write Time : " << readWriteTime.toString() << std::endl;
	std::cout << "		GPU Execution Time : " << gpuExecutionTime.toString() <<"\n"<< std::endl;

	return gpuExecutionTime;
}

