
include_directories(
  model
)

add_executable(NNBiRNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiRNNLabeler ${LIBS})
