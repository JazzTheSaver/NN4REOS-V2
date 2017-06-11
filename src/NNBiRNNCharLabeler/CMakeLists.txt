
include_directories(
  model
)

add_executable(NNBiRNNCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiRNNCharLabeler ${LIBS})
