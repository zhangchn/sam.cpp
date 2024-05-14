set(SAM_TARGET sam_cli)

INCLUDE_DIRECTORIES(
    ../examples
)
add_executable(${SAM_TARGET}
    cli.cpp
    ../examples/stb_image.h
    ../examples/stb_image_write.h
)

install(TARGETS ${SAM_TARGET} RUNTIME)
target_link_libraries(${SAM_TARGET} PRIVATE sam.cpp  ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SAM_TARGET} PUBLIC cxx_std_11)

