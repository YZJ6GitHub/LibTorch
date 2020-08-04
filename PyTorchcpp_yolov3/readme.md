Darknet.h  :  head file

Darknet :   network file

main : yolov3 used by C++


necessary file:

      yolov3.weights,you can down on the Internet. push the file into the models direction.


1. Create new file by vs2019 and set up environment 
   
  -------------include:----------------------

     D:\opencv4.0\build\include
     D:\opencv4.0\build\include\opencv2
     D:\libtorch\debug\include
     D:\libtorch\debug\include\torch\csrc\api\include

   -------------lib:----------------------

     D:\libtorch\debug\lib
     D:\opencv4.0\build\x64\vc15\lib

    -------------link:----------------------

    opencv_world400d.lib
    torch.lib
    c10.lib
    c10_cuda.lib
    caffe2_module_test_dynamic.lib
    caffe2_nvrtc.lib
    clog.lib
    cpuinfo.lib
    libprotobufd.lib
    libprotobuf-lited.lib
    libprotocd.lib
     
 2. Test the environment
 
   
  1)load head file:
     
       #include <torch/torch.h>
       #include <iostream>
       
       
  2)Test context:
   
    int main()
    {
         
         torch::Tensor tensor = torch::rand({2, 3});
         std::cout << tensor << std::endl;        
         return 0
    }



 3 . run main.cpp 
