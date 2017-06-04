
include_directories(
  model
)

add_executable(NNGRNNCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNGRNNCharLabeler ${LIBS})
