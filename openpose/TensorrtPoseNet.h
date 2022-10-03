#pragma once
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include "NvInfer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <NvOnnxParser.h>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include "resize.h"

enum class RUN_MODE
{
	FLOAT32 = 0,
	FLOAT16 = 1,
	INT8 = 2
};

struct InferDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};


inline void* safeCudaMalloc(size_t memSize)
{
	void* deviceMem;
	cudaMalloc(&deviceMem, memSize);
	if (deviceMem == nullptr)
	{
		std::cerr << "Out of memory" << std::endl;
		exit(1);
	}
	return deviceMem;
}



class TensorrtPoseNet
{
	template <typename T>
	using UniquePtr = std::unique_ptr<T, InferDeleter>;

public: 
	TensorrtPoseNet(const std::string &engineFilePath = "trt_pose_fp16.engine", float confThresh = 0.01, float nmsThresh = 0.1);
	
	void infer(cv::Mat &img);	

	bool saveEngine(const std::string &engineFilePath);
	~TensorrtPoseNet();

	// The dimensions of the input and output to the network

	int batchSize;
	int numClasses;
	int numChannels;
	int inputHeightSize;
	int inputWidthSize;
	
	std::vector<float> cpuCmapBuffer;
	std::vector<float> cpuPafBuffer;

	std::vector<nvinfer1::Dims> inputDims;
	std::vector<nvinfer1::Dims> outputDims;


private:
	std::size_t getSizeByDim(const nvinfer1::Dims& dims);
	void preprocessImage(cv::Mat &frame, float* gpu_input);

	void initEngine();

	UniquePtr<nvinfer1::ICudaEngine> engine;
	UniquePtr<nvinfer1::IExecutionContext> context;
	cudaStream_t cudaStream;
	
	std::vector<void*> cudaBuffers;
	void *cudaFrame;

	float confThreshold = 0.01f;
	float nmsThreshold = 0.1f;


};