#/bin/tcsh

#setenv OpenCL_LIBPATH /opt/intel/opencl-1.2-3.0.56860/lib

/usr/pic1/cmake-3.4.0-Linux-x86_64/bin/cmake \
  -DICC_LIB_LOCATION=/rel/third_party/intelcompiler64/composer_xe/composer_xe_2013_sp1.3.174/compiler/lib        \
  -DCMAKE_C_COMPILER=/rel/third_party/intelcompiler64/composer_xe/composer_xe_2013_sp1.3.174/bin/intel64/icc     \
  -DCMAKE_CXX_COMPILER=/rel/third_party/intelcompiler64/composer_xe/composer_xe_2013_sp1.3.174/bin/intel64/icpc  \
  -DTBB_LOCATION=/rel/third_party/intelcompiler64/composer_xe/composer_xe_2013_sp1.3.174/tbb                     \
  -DISPC_LOCATION=/rel/third_party/intelispc/1.9.1                                                               \
  -DGLEW_LOCATION=/usr/pic1/glew-1.9.0                                                                           \
  -DGLFW_LOCATION=/usr/pic1/glfw-3.0.2                                                                           \
  -DCMAKE_BUILD_TYPE=Release                                                                                   \
  -DNO_PYTHON=1                                                                                                  \
  -DEMBREE_LOCATION=/usr/pic1/embree/install-avx/  \
  ..
  
