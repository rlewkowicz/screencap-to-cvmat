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

#include <ecal/ecal.h>
#include <ecal/msg/protobuf/publisher.h>

#include <iostream>
#include <thread>
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
#include <sstream>
#include <cstring> 
#include "mouse.pb.h"

#include <cstdlib>

eCAL::protobuf::CPublisher<proto_messages::mouse_report> publisher;

cv::Mat mat;
bool DEBUG=false;

struct img_and_coord {
    cv::Mat mat;
    int x = 999;
    int y = 999;
    int conf;
};

typedef struct img_and_coord img_and_coord;

img_and_coord last_img_and_coord;
int global_conf = 0;
int frames_without_detect = 3;
int last_conf = 0;
int divisor = 1;

using namespace std;
using namespace cv;

TensorrtPoseNet posenet;
Openpose openpose(posenet.outputDims[0]);

static const char* cocolabels[] = {
    "enemy"
};

std::string gen_random(const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    std::string tmp_s;
    tmp_s.reserve(len);

    for (int i = 0; i < len; ++i) {
        tmp_s += alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    return tmp_s;
}

int COUNT = 0;
static img_and_coord run_inf_debug(shared_ptr<Yolo::Infer> engine, cv::Mat image, int deviceid, TRT::Mode mode, Yolo::Type type, bool is_ret = false) {
    img_and_coord img_and_coord;
    const int ntest = 100;
    vector<shared_future<Yolo::BoxArray>> boxes_array;
    boxes_array.clear();
    vector<cv::Mat> images;

    COUNT = COUNT + 1;

    images.emplace_back(image);
    boxes_array = engine->commits(images);
    boxes_array.back().get();

    auto type_name = Yolo::type_name(type);
    auto mode_name = TRT::mode_string(mode);

    for (int i = 0; i < 1; ++i) {
        auto boxes = boxes_array[i].get();

        for (auto& obj : boxes) {
            if (COUNT > 400) {
                if (obj.confidence * 100 < 80 && obj.confidence * 100 > 10) {
                    if (is_ret == false) {
                        cv::imwrite("C:\\tmp\\guard\\" + gen_random(14) + ".bmp", image);
                    }
                    else {
                        cv::imwrite("C:\\tmp\\ret\\" + gen_random(14) + ".bmp", image);
                    }
                }
                COUNT = 0;
            }
            if (false) {
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
            img_and_coord.conf = int(obj.confidence*100);
        }
        if (DEBUG) {
            img_and_coord.mat = image;
        }

        return img_and_coord;
    }
}

static shared_ptr<Yolo::Infer> app_yolo(Yolo::Type type, TRT::Mode mode, const string& model, string prefix) {

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {

        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            Yolo::image_to_tensor(image, tensor, type, i);
        }
    };
    string onnx_string = prefix + ".onnx";
    string model_string = prefix + ".trtmodel";
    string onnx_file = iLogger::format(onnx_string.c_str());
    string model_file = iLogger::format(model_string.c_str());
    int test_batch_size = 1;

    if (!iLogger::exists(model_file)) {
        TRT::compile(
            mode,                       // FP16、FP16、INT8
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
    float thresh;
    if (onnx_string == "ret.onnx") {
        thresh = 0.3f;
    }
    else {
        thresh = 0.2f;
    }
    auto engine = Yolo::create_infer(
        model_file,                // engine file
        type,                       // yolo type, Yolo::Type::V5 / Yolo::Type::X
        deviceid,                   // gpu id
        thresh,                // confidence threshold
        0.90f,                      // nms threshold
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

    img_and_coord determine_confidence(img_and_coord img_and_coord) {
        global_conf = img_and_coord.conf;
        int x = img_and_coord.x;
        int y = img_and_coord.y;
        int last_x = last_img_and_coord.x;
        int last_y = last_img_and_coord.y;
        int my_count = 0;
        for (auto i : pose_vec) {
            int is_even;
            for (auto a : i) {
                if (is_even == 0) {
                    is_even = is_even + 1;
                    if (abs(x - a) < 30) {
                        my_count = my_count + 1;
                    }
                }
                else {
                    if (abs(y - a) < 30) {
                        my_count = my_count + 1;
                    }
                }
            }
        }
        pose_vec.clear();
        //cout << my_count << endl;
        if (my_count > 1) {
            global_conf += 59;
        }
        my_count = 0;
        if (last_conf < 0)
            last_conf = 0;
        if (last_conf > 70) {
            if (abs(x - last_x) < 20 && abs(y - last_y) < 20) {
                global_conf += 39;
            }
            else if (abs(x - last_x) < 30 && abs(y - last_y) < 30) {
                global_conf += 29;
            }
        }
        else if (last_conf != 0) {
            img_and_coord.x = (last_x + x + x)/3;
            img_and_coord.y = (last_y + y + y)/3;
        }
        last_conf = global_conf;
        last_img_and_coord = img_and_coord;
        my_count = 0;
        return img_and_coord;
    }

    static int run_pose(int i) {
        posenet.infer(mat);
        openpose.detect(posenet.cpuCmapBuffer, posenet.cpuPafBuffer, mat);
        return 0;
    }

    static int find_red(cv::Mat mat) {
        bool found = false;
        int rgb_total = 0;
        int rgbs_count = 0;
        int gap = 0;
        bool grey = false;
        bool white = false;

        for (int x = 0; x < 640; x++) {
            for (int y = 0; y < 640; y++) {
                int b = mat.at<Vec3b>(y, x)[0];
                int g = mat.at<Vec3b>(y, x)[1];
                int r = mat.at<Vec3b>(y, x)[2];
                if (grey == true && (r > 140 && r < 220) && (g > 40 && g < 80) && (b > 30 && b < 130)) {
                    white == false;

                    for (int i = 4; i < 8; i++) {
                        if (y - i < 640 && y - i > 0) {
                            int b = mat.at<Vec3b>(y - i, x)[0];
                            int g = mat.at<Vec3b>(y - i, x)[1];
                            int r = mat.at<Vec3b>(y - i, x)[2];

                            if (r > 175 && g > 175 && b > 175) {
                                white = true;
                                break;
                            }
                        }
                    }

                    if (white == true) {
                        for (int i = 0; i < 40; i++) {
                            if (y + i < 640 && y + i > 0) {
                                int b = mat.at<Vec3b>(y + i, x)[0];
                                int g = mat.at<Vec3b>(y + i, x)[1];
                                int r = mat.at<Vec3b>(y + i, x)[2];


                                if ((r > 140 && r < 220) && (g > 40 && g < 80) && (b > 30 && b < 130)) {
                                    rgb_total = rgb_total + 1;
                                }
                                else if (rgb_total > 0) {
                                    gap = gap + 1;
                                }
                                if (gap > 0) {
                                    if (rgb_total < 7 || rgb_total > 20) {
                                        gap = 0;
                                        rgb_total = 0;
                                    }
                                    else if (rgb_total > 7 && rgb_total < 20) {
                                        for (int a = 0; a < rgb_total; a++) {
                                            if (y + a < 640 && y + a > 0) {
                                                //mat.at<Vec3b>(y + a, x)[0] = 255;
                                                //mat.at<Vec3b>(y + a, x)[1] = 255;
                                                //mat.at<Vec3b>(y + a, x)[2] = 255;
                                            }
                                        }
                                        found = true;
                                        rgbs_count = rgbs_count + 1;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (rgb_total > 5 && rgb_total < 25) {
                        found = true;
                        break;
                    }
                }
                else if (grey == false && ((r > 40 && r < 130) && (g > 40 && g < 130) && (b > 40 && b < 130))) {
                    for (int i = 0; i < 20; i++) {
                        if (y + i < 640 && y + i > 0) {
                            int b = mat.at<Vec3b>(y + i, x)[0];
                            int g = mat.at<Vec3b>(y + i, x)[1];
                            int r = mat.at<Vec3b>(y + i, x)[2];


                            if (r > 200 && g > 200 && b > 200) {
                                grey = true;
                            }
                        }
                    }
                }
                if (found) {
                    found = false;
                    break;
                }
            }
        }
        return rgbs_count;
    }

    int pose_count = 2;
    HRESULT Preproc(shared_ptr<Yolo::Infer> engine, Yolo::Type type, TRT::Mode mode, const string& model, shared_ptr<Yolo::Infer> ret)
    {
        int arr[2] = { 0,0 };

        HRESULT hr = S_OK;

        D3D11_TEXTURE2D_DESC desc;
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

        img_and_coord ret_coord;
        img_and_coord yolo;
        int rgb_total = 0;

        if (pose_count == 2) {
            auto t4 = std::async(std::launch::async, find_red, mat);
            auto t3 = std::async(std::launch::async, run_pose, 1);
            auto t1 = std::async(std::launch::async, run_inf_debug, engine, mat, 1, mode, type, false);
            auto t2 = std::async(std::launch::async, run_inf_debug, ret, mat, 1, TRT::Mode::FP16, Yolo::Type::V5, true);
            ret_coord = t2.get();
            yolo = t1.get();
            auto a = t3.get();
            rgb_total = t4.get();
            pose_count = 0;
        }
        else {
            auto t4 = std::async(std::launch::async, find_red, mat);
            auto t1 = std::async(std::launch::async, run_inf_debug, engine, mat, 1, mode, type, false);
            auto t2 = std::async(std::launch::async, run_inf_debug, ret, mat, 1, TRT::Mode::FP16, Yolo::Type::V5, true);
            ret_coord = t2.get();
            yolo = t1.get();
            rgb_total = t4.get();
            pose_count = pose_count + 1;
        }

        if (yolo.x == 999 || yolo.y == 999) {
            frames_without_detect = frames_without_detect + 1;
            if (frames_without_detect >= 3) {
                frames_without_detect = 3;
                global_conf = 0;
                last_conf = 0;
            }
        }
        else {
            frames_without_detect = 0;
        }

        if (yolo.x != 999 && yolo.y != 999) {

            yolo = determine_confidence(yolo);

            int x = int(-(320 - yolo.x));
            int y = int(-(320 - yolo.y));
            int dx = 0;
            int dy = 0;
            if (ret_coord.x != 999 || ret_coord.y != 999) {
                if (ret_coord.conf > 1) {
                    dx = int((ret_coord.x - yolo.x));
                    dy = int((ret_coord.y - yolo.y));
                }

                x = -dx+20;
                y = -dy+20;

            }
            proto_messages::mouse_report mouse_report;
            mouse_report.set_x(x);
            mouse_report.set_y(y);

            if (rgb_total > 6 && rgb_total < 200) {
                global_conf = global_conf + 40;
            }
            else {
                global_conf = global_conf - 30;
            }

            if (global_conf > 68) {
                 publisher.Send(mouse_report);
            }

            
            SAFE_RELEASE(pDupTex2D);
            SAFE_RELEASE(myText);
        }            
        
        if (DEBUG) {
            if (yolo.x != 999 || yolo.y != 999) {
                cout << "conf: " << yolo.conf << endl;
                cout << "global_conf: " << global_conf << endl << endl;
            }
            cv::namedWindow("enemy");
            cv::resizeWindow("enemy", 640, 640);
            cv::imshow("enemy", ret_coord.mat);
            //cv::imshow("enemy", mat);
            cv::waitKey(1);
        }
        SAFE_RELEASE(pDupTex2D);
        SAFE_RELEASE(myText);
        returnIfError(hr);

        return hr;
        pose_vec.clear();
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

int frame_avg_num = 0;
int frame_avg_total = 0;
int Grab60FPS(int nFrames)
{
    if (DEBUG == true) {
        iLogger::set_log_level(iLogger::LogLevel::Debug);
    }
    else {
        iLogger::set_log_level(iLogger::LogLevel::Fatal);
    }
    string yolo_mode = "yolov5x6";
    auto t1 = std::async(std::launch::async, app_yolo, Yolo::Type::V5, TRT::Mode::FP16, "yolov5s6", "x6");
    auto t2 = std::async(std::launch::async, app_yolo, Yolo::Type::V5, TRT::Mode::FP16, "yolov5s6", "ret");
    auto engine = t1.get();
    auto ret = t2.get();
    //auto engine = app_yolo(Yolo::Type::V5, TRT::Mode::FP16, "yolov5s6", "x6");
    //auto ret = app_yolo(Yolo::Type::V5, TRT::Mode::FP16, "yolov5s6", "ret");
    const int WAIT_BASE = 500;
    DemoApplication Demo;
    HRESULT hr = S_OK;
    int capturedFrames = 0;
    LARGE_INTEGER start = { 0 };
    LARGE_INTEGER end = { 0 };
    LARGE_INTEGER interval = { 0 };
    LARGE_INTEGER freq = { 0 };
    int wait = WAIT_BASE;

  //  QueryPerformanceFrequency(&freq);

    /// Reset waiting time for the next screen capture attempt
/*#define RESET_WAIT_TIME(start, end, interval, freq)         \
    QueryPerformanceCounter(&end);                          \
    interval.QuadPart = end.QuadPart - start.QuadPart;      \
    MICROSEC_TIME(interval, freq);    \  */                    

    hr = Demo.Init();
    if (FAILED(hr))
    {
        printf("Initialization failed with error 0x%08x\n", hr);
        return -1;
    }

    do
    {

        if (frame_avg_num < 3) {
        auto start = std::chrono::high_resolution_clock::now();
        //QueryPerformanceCounter(&start);
        hr = Demo.Capture(wait);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) 
        {
           // RESET_WAIT_TIME(start, end, interval, freq);
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
               // RESET_WAIT_TIME(start, end, interval, freq);
               // QueryPerformanceCounter(&start);
                Demo.Capture(wait);
            }
            //RESET_WAIT_TIME(start, end, interval, freq);


                hr = Demo.Preproc(engine, Yolo::Type::V5, TRT::Mode::FP16, yolo_mode, ret);

            }

            if (FAILED(hr))
            {
                printf("Preproc failed with error 0x%08x\n", hr);
                return -1;
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            frame_avg_total = frame_avg_total + duration.count();
            frame_avg_num = frame_avg_num + 1;
        }
    else {
        //cout << frame_avg_total / 3 << endl;
        frame_avg_total = 0;
        frame_avg_num = 0;
    }

    } while (true);

    return 0;
}


int main(int argc, char** argv)
{
    eCAL::Initialize(argc, argv, "yolo");
    publisher = eCAL::protobuf::CPublisher<proto_messages::mouse_report>("yolo");

    DEBUG = false;
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