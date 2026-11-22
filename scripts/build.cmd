@echo off
rem Build ggml-base + ggml-cuda from llama.cpp\src (MSVC + nvcc 13.5, sm_120a only), mirroring the official windows-cuda release job.
rem Usage: scripts\build.cmd <outdir-name>   e.g. scripts\build.cmd vanilla   -> llama.cpp\build-vanilla, llama.cpp\bin-vanilla
setlocal
set ROOT=%~dp0..
set NAME=%1
if "%NAME%"=="" set NAME=dev
set CUDA_PATH=%ROOT%\toolchain\cuda-13.5
set PATH=%ROOT%\.venv\Scripts;%CUDA_PATH%\bin;%PATH%
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul || exit /b 1
cmake -S "%ROOT%\llama.cpp\src" -B "%ROOT%\llama.cpp\build-%NAME%" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF -DGGML_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120a-real -DCUDAToolkit_ROOT="%CUDA_PATH%" -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF || exit /b 1
cmake --build "%ROOT%\llama.cpp\build-%NAME%" --config Release -j 15 --target ggml-base ggml-cuda llama || exit /b 1
rem assemble: prebuilt b12615 folder with our ggml-base.dll + ggml-cuda.dll swapped in
robocopy "%ROOT%\llama.cpp\b12615" "%ROOT%\llama.cpp\bin-%NAME%" /E /NFL /NDL /NJH /NJS >nul
copy /Y "%ROOT%\llama.cpp\build-%NAME%\bin\ggml-base.dll" "%ROOT%\llama.cpp\bin-%NAME%\" || exit /b 1
copy /Y "%ROOT%\llama.cpp\build-%NAME%\bin\ggml-cuda.dll" "%ROOT%\llama.cpp\bin-%NAME%\" || exit /b 1
copy /Y "%ROOT%\llama.cpp\build-%NAME%\bin\llama.dll" "%ROOT%\llama.cpp\bin-%NAME%\" || exit /b 1
echo BUILD OK: %ROOT%\llama.cpp\bin-%NAME%
