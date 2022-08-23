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

    HRESULT Preproc()
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
        cv::UMat mat;
        cv::directx::convertFromD3D11Texture2D(myText, mat);
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2RGB);

        cv::namedWindow("enemy", cv::WINDOW_AUTOSIZE);
        cv::imshow("enemy", mat);
        cv::waitKey(1);

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
            hr = Demo.Preproc(); 
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
    int nFrames = 1;
    int ret = 0;
    
    ret = Grab60FPS(nFrames);
    return ret;
}