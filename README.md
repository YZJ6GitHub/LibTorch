# PyTorchCPP
C++ libPyTorch the training result of PyTorch is  used by C++ 


train_minist  :  In order to train model on minist datasets, the type of mode is .pth.

test_minist :  Input n new image to predict what type is belongs to.

transform_model : In order to make use of the training model ,must transoform the type of .pth  to pt. the file can achieve it.

main : the file is to realize the model of PyTorch to used in C++ Program 



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

    opencv_world400d.lib;torch_cpu.lib;torch.lib;c10.lib
     
 2. Test the environment

    #include <torch/torch.h>
    #include <iostream>
     int main()
     {
         
         torch::Tensor tensor = torch::rand({2, 3});
         std::cout << tensor << std::endl;        
         return 0
     }
   
 
      

 



     

