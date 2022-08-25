
#if defined(_WIN32)
#	define U_OS_WINDOWS
#else
#   define U_OS_LINUX
#endif

#ifdef U_OS_WINDOWS
#if defined(_DEBUG)
#else
#endif

//导入cuda
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "cublas.lib")
#pragma comment(lib, "cudnn.lib")

//导入tensorRT
#pragma comment(lib, "nvinfer.lib")
#pragma comment(lib, "nvinfer_plugin.lib")
#pragma comment(lib, "nvparsers.lib")

#if defined(_DEBUG)
#pragma comment(lib, "libprotobufd.lib")
#else
#pragma comment(lib, "libprotobuf.lib")
#endif

#ifdef HAS_PYTHON
#pragma comment(lib, "python37.lib")
#endif

#endif // U_OS_WINDOWS