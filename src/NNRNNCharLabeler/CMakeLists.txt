
include_directories(
  model
)

add_executable(NNRNNCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNRNNCharLabeler ${LIBS})
