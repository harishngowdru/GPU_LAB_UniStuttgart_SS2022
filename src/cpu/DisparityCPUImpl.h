#include <vector>

void DisparityMappingCPUImpl(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& h_outputSSD, size_t countX, size_t countY, int isSSD);