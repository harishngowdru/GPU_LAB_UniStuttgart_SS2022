#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>
#include <vector>

struct KernelParameter {
	std::size_t wgSizeX;	// Number of work items per work group in X direction
	std::size_t wgSizeY;
	std::size_t countX;		// Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY;
	std::size_t count;		// Overall number of elements
	std::size_t size;		// Size of data in bytes
};

class OpenCLConfig{
public:
	bool createPlatform();
	void createContext();
	cl::Context getContext();
	void setKernelParameter(KernelParameter param);
	KernelParameter getKernelParameter();
	void createCommandQueue(int deviceNumber);
	void setCLFilePath(std::string path);
	Core::TimeSpan executeKernel(std::vector<float>& outputGpuSSD, std::vector<float>& inputLeftImage, std::vector<float>& inputRightImage, bool isSSD);

private:
	std::vector<cl::Platform> platforms;
	int platformId = 0;
	cl::Context context;
	KernelParameter kernelParameter;
	
	cl::Device device;
	std::vector<cl::Device> devices;
	cl::CommandQueue queue;

	std::string  clFilePath;
};

