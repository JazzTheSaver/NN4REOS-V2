
include_directories(
  model
)

add_executable(NNBiGRNNCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiGRNNCharLabeler ${LIBS})
