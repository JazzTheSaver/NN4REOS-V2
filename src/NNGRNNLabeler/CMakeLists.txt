
include_directories(
  model
)

add_executable(NNGRNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNGRNNLabeler ${LIBS})
