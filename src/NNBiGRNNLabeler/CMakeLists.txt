
include_directories(
  model
)

add_executable(NNBiGRNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiGRNNLabeler ${LIBS})
