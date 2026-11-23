# the policy header from cpp/ (one copy); model.bin and trace.bin from export_cpp.py --out embedded/esp32/main
idf_component_register(SRCS "main.cpp" "dsp_ansi.c" INCLUDE_DIRS "." "../../../cpp" EMBED_FILES "model.bin" "trace.bin")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++17>)
