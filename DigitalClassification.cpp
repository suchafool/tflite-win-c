#include "opencv2/opencv.hpp"
#include "DigitalClassification.h"

DigitalClassification::DigitalClassification(const char* modelPath)
{
	initModel(modelPath);
}

DigitalClassification::~DigitalClassification()
{
	if (m_model != nullptr) 
	{
		TfLiteModelDelete(m_model);
	}
}

int DigitalClassification::recognize(Mat src)
{
	Mat image;
	cv::resize(src, image, cv::Size(28, 28), cv::INTER_NEAREST);
	image.convertTo(image, CV_32FC3, 1/255.0);

	float* dst = m_input_tensor->data.f;
	memcpy(dst, image.data, sizeof(float)*image.total() );


	if (TfLiteInterpreterInvoke(m_interpreter) != kTfLiteOk) 
	{
		printf("Error invoking classification model.");
		return -1;
	}

	const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(m_interpreter, 0);

	float result[10];
	TfLiteTensorCopyToBuffer(output_tensor, result, sizeof(float)*10);

	int maxIndex = 0;

	for (int i = 0; i < 10; i++) {
		if (result[i] > result[maxIndex])
		{
			maxIndex = i;
		}
	}

	printf("Max confidence is: %.6f, at: %d. \n", result[maxIndex], maxIndex);

	return maxIndex;
}

void DigitalClassification::initModel(const char* modelPath)
{
	m_model = TfLiteModelCreateFromFile(modelPath);

	// Build the interpreter
	TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 1);

	// Create the interpreter.
	m_interpreter = TfLiteInterpreterCreate(m_model, options);
	if (m_interpreter == nullptr) {
		printf("Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (TfLiteInterpreterAllocateTensors(m_interpreter) != kTfLiteOk) {
		printf("Failed to allocate tensors!");
		return;
	}

	// Find input tensors.
	if (TfLiteInterpreterGetInputTensorCount(m_interpreter) != 1) {
		printf("Classification model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = TfLiteInterpreterGetInputTensor(m_interpreter, 0);
	if (m_input_tensor->type != kTfLiteFloat32) {
		printf("Classification model input should be kTfLiteFloat32!");
		return;
	}

	if (TfLiteInterpreterGetOutputTensorCount(m_interpreter) != 1) {
		printf("Classification model graph needs to have 1 and only 1 output!");
		return;
	}
}
