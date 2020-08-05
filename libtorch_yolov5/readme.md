      
1. environment


   vs 2017 + 
  
   opencv 4.0
   
   libtorch 1.6
   
   
   
2. Configuration Environment


   include ：
   D:\opencv4.0\build\include
 
   D:\opencv4.0\build\include\opencv2
   
   D:\libtorch\debug\include
   
   D:\libtorch\debug\include\torch\csrc\api\include
   
   
   
   lib:
   D:\libtorch\debug\lib
   
   D:\opencv4.0\build\x64\vc15\lib
   
   link:
   
   opencv_world400d.lib
   
   torch_cpu.lib
   
   torch.lib
   
   c10.lib
   

3. Run main


4. file Explain:
   
   images : the input images which need to inference by model
   
   
   models : the model is transformed by export.py,which can be used by C++ directly
   
   
   weights : the label name
   
   
   utils : the file of transform model 
   
   
   Result : inference result
   

 
