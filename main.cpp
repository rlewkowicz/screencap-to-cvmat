/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Defs.h"
#include "DDAImpl.h"
#include "opencv2/core/directx.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "tensorRT/builder/trt_builder.hpp"
#include "tensorRT/infer/trt_infer.hpp"
#include "tensorRT/common/ilogger.hpp"
#include "yolo/yolo.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/opengl.hpp"
#include <opencv2/highgui/highgui_c.h>
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "openpose/Openpose.h"
#include "openpose/TensorrtPoseNet.h"
#include <atlstr.h>
#include <future>

cv::Mat mat;
bool DEBUG=false;

using namespace std;
using namespace cv;

TensorrtPoseNet posenet;
Openpose openpose(posenet.outputDims[0]);

static const char* cocolabels[] = {
    "enemy"
};

struct img_and_coord {
    cv::Mat mat;
    int x = 999;
    int y = 999;
};

typedef struct img_and_coord img_and_coord;

static img_and_coord run_inf_debug(shared_ptr<Yolo::Infer> engine, cv::Mat image, int deviceid, TRT::Mode mode, Yolo::Type type) {
    img_and_coord img_and_coord;
    const int ntest = 100;
    vector<shared_future<Yolo::BoxArray>> boxes_array;
    boxes_array.clear();
    vector<cv::Mat> images;

    images.emplace_back(image);
    boxes_array = engine->commits(images);
    boxes_array.back().get();

    auto type_name = Yolo::type_name(type);
    auto mode_name = TRT::mode_string(mode);

    for (int i = 0; i < 1; ++i) {

        auto boxes = boxes_array[i].get();

        for (auto& obj : boxes) {
            if (DEBUG) {
                uint8_t b, g, r;
                tie(b, g, r) = iLogger::random_color(obj.class_label);
                cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

                auto name = cocolabels[obj.class_label];
                auto caption = iLogger::format("%s %.2f", name, obj.confidence);
                int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            img_and_coord.x = obj.right;
            img_and_coord.y = obj.bottom;
        }
        if (DEBUG) {
            img_and_coord.mat = image;
        }

        return img_and_coord;
    }
}

static shared_ptr<Yolo::Infer> app_yolo(Yolo::Type type, TRT::Mode mode, const string& model) {

    int deviceid = 1;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {

        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };

    string onnx_file = iLogger::format("best.onnx");
    string model_file = iLogger::format("best.trtmodel");
    int test_batch_size = 1;


    if (!iLogger::exists(model_file)) {
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference",
            "inference",
            10000000000
        );
    }

    auto engine = Yolo::create_infer(
        model_file,                // engine file
        type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        deviceid,                   // gpu id
        0.60f,                      // confidence threshold
        0.80f,                      // nms threshold
        Yolo::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1,                          // max objects
        false                       // preprocess use multi stream
    );
    if (engine == nullptr) {
        INFOE("Engine is nullptr");
        exit(1);
    }

    return engine;
}

class DemoApplication
{
#define returnIfError(x)\
    if (FAILED(x))\
    {\
        printf(__FUNCTION__": Line %d, File %s Returning error 0x%08x\n", __LINE__, __FILE__, x);\
        return x;\
    }

private:
    DDAImpl *pDDAWrapper = nullptr;
    ID3D11Device *pD3DDev = nullptr;
    ID3D11DeviceContext *pCtx = nullptr;
    ID3D11Texture2D* pDupTex2D = nullptr;
    ID3D11Texture2D* myText = nullptr;
    ID3D11Texture2D *pEncBuf = nullptr;
    FILE *fp = nullptr;
    UINT failCount = 0;
    const static bool bNoVPBlt = false;
    const char fnameBase[64] = "DDATest_%d.h264";
    std::vector<std::vector<uint8_t>> vPacket;
    D3D11_BOX my_box;

private:
    HRESULT InitDXGI()
    {
        HRESULT hr = S_OK;
        D3D_DRIVER_TYPE DriverTypes[] =
        {
            D3D_DRIVER_TYPE_HARDWARE,
            D3D_DRIVER_TYPE_WARP,
            D3D_DRIVER_TYPE_REFERENCE,
        };
        UINT NumDriverTypes = ARRAYSIZE(DriverTypes);

        D3D_FEATURE_LEVEL FeatureLevels[] =
        {
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_1
        };
        UINT NumFeatureLevels = ARRAYSIZE(FeatureLevels);
        D3D_FEATURE_LEVEL FeatureLevel = D3D_FEATURE_LEVEL_11_0;

        for (UINT DriverTypeIndex = 0; DriverTypeIndex < NumDriverTypes; ++DriverTypeIndex)
        {
            hr = D3D11CreateDevice(nullptr, DriverTypes[DriverTypeIndex], nullptr, /*D3D11_CREATE_DEVICE_DEBUG*/0, FeatureLevels, NumFeatureLevels,
                D3D11_SDK_VERSION, &pD3DDev, &FeatureLevel, &pCtx);
            if (SUCCEEDED(hr))
            {
                break;
            }
        }
        return hr;
    }

    HRESULT InitDup()
    {
        my_box.front = 0;
        my_box.back = 1;
        my_box.left = 1600;
        my_box.top = 480;
        my_box.right = 2240;
        my_box.bottom = -160;

        cv::directx::ocl::initializeContextFromD3D11Device(pD3DDev);

        HRESULT hr = S_OK;
        if (!pDDAWrapper)
        {
            pDDAWrapper = new DDAImpl(pD3DDev, pCtx);
            hr = pDDAWrapper->Init();
            returnIfError(hr);
        }
        return hr;
    }

public:
    HRESULT Init()
    {
        HRESULT hr = S_OK;

        hr = InitDXGI();
        returnIfError(hr);

        hr = InitDup();
        returnIfError(hr);

        return hr;
    }

    HRESULT Capture(int wait)
    {
        HRESULT hr = pDDAWrapper->GetCapturedFrame(&pDupTex2D, wait); // Release after preproc
        if (FAILED(hr))
        {
            failCount++;
        }
        return hr;
    }

    void run_pose() {
        posenet.infer(mat);
        openpose.detect(posenet.cpuCmapBuffer, posenet.cpuPafBuffer, mat);
    }

    HRESULT Preproc(shared_ptr<Yolo::Infer> engine, Yolo::Type type, TRT::Mode mode, const string& model)
    {
        HRESULT hr = S_OK;

        D3D11_TEXTURE2D_DESC desc;
        ZeroMemory(&desc, sizeof(desc));
        pDupTex2D->GetDesc(&desc);

        desc.Width = 640;
        desc.Height = 640;
        desc.ArraySize = 1;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.MipLevels = 1;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.Usage = D3D11_USAGE_STAGING;

        hr = pD3DDev->CreateTexture2D(&desc, NULL, &myText);
        pCtx->CopySubresourceRegion(myText, D3D11CalcSubresource(0, 0, 1), 0, 0, 0, pDupTex2D, 0, &my_box);
        
        ZeroMemory(&mat, sizeof(mat));

        cv::directx::convertFromD3D11Texture2D(myText, mat);

        cv::cvtColor(mat, mat, cv::COLOR_RGBA2RGB);
            
        auto t1 = std::async(std::launch::async, run_inf_debug, engine, mat, 1, mode, type);
        run_pose();
        img_and_coord img_and_coord = t1.get();
        
        if (DEBUG) {
            cv::namedWindow("enemy");
            cv::resizeWindow("enemy", 640, 640);
            cv::imshow("enemy", img_and_coord.mat);
            cv::waitKey(1);
        }
        SAFE_RELEASE(pDupTex2D);
        returnIfError(hr);

        return hr;
    }

    void Cleanup(bool bDelete = true)
    {
        if (pDDAWrapper)
        {
            pDDAWrapper->Cleanup();
            delete pDDAWrapper;
            pDDAWrapper = nullptr;
        }

        SAFE_RELEASE(pDupTex2D);
        if (bDelete)
        {        
            SAFE_RELEASE(pD3DDev);
            SAFE_RELEASE(pCtx);
        }
    }
    DemoApplication() {}
    ~DemoApplication()
    {
        Cleanup(true); 
    }
};

int Grab60FPS(int nFrames)
{
    iLogger::set_log_level(iLogger::LogLevel::Debug);
    auto engine = app_yolo(Yolo::Type::V5, TRT::Mode::FP16, "yolov5l");
    const int WAIT_BASE = 6;
    DemoApplication Demo;
    HRESULT hr = S_OK;
    int capturedFrames = 0;
    LARGE_INTEGER start = { 0 };
    LARGE_INTEGER end = { 0 };
    LARGE_INTEGER interval = { 0 };
    LARGE_INTEGER freq = { 0 };
    int wait = WAIT_BASE;

    QueryPerformanceFrequency(&freq);

    /// Reset waiting time for the next screen capture attempt
#define RESET_WAIT_TIME(start, end, interval, freq)         \
    QueryPerformanceCounter(&end);                          \
    interval.QuadPart = end.QuadPart - start.QuadPart;      \
    MICROSEC_TIME(interval, freq);                          \

    hr = Demo.Init();
    if (FAILED(hr))
    {
        printf("Initialization failed with error 0x%08x\n", hr);
        return -1;
    }

    do
    {
        QueryPerformanceCounter(&start);
        hr = Demo.Capture(wait);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) 
        {
            RESET_WAIT_TIME(start, end, interval, freq);
            continue;
        }
        else
        {
            if (FAILED(hr))
            {
                printf("Captrue failed with error 0x%08x. Re-create DDA and try again.\n", hr);
                Demo.Cleanup();
                hr = Demo.Init();
                if (FAILED(hr))
                {
                    printf("Failed to Init DDDemo. return error 0x%08x\n", hr);
                    return -1;
                }
                RESET_WAIT_TIME(start, end, interval, freq);
                QueryPerformanceCounter(&start);
                Demo.Capture(wait);
            }
            RESET_WAIT_TIME(start, end, interval, freq);
            hr = Demo.Preproc(engine, Yolo::Type::V5, TRT::Mode::FP16, "yolov5l");
            if (FAILED(hr))
            {
                printf("Preproc failed with error 0x%08x\n", hr);
                return -1;
            }
        }
    } while (true);

    return 0;
}

int main(int argc, char** argv)
{

    CString envvar;
    if (envvar.GetEnvironmentVariable(_T("DEBUG")))
    {
        DEBUG = true;
    }

    int nFrames = 1;
    int ret = 0;
    
    ret = Grab60FPS(nFrames);
    return ret;
}