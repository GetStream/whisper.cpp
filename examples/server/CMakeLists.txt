cmake_minimum_required(VERSION 3.1)
project(whisper_server)

set(TARGET server)

# Find Threads package if not already found
find_package(Threads REQUIRED)

add_executable(${TARGET}
    server.cpp
    thread_pool.cpp
    thread_pool.h
    httplib.h
)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE
    common
    json_cpp
    whisper
    ${CMAKE_THREAD_LIBS_INIT}
)

if(WIN32)
    target_link_libraries(${TARGET} PRIVATE ws2_32)
endif()
