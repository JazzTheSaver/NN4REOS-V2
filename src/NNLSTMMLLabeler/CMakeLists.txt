
include_directories(
  model
)

add_executable(NNLSTMMLLabeler NNCNNLabeler.cpp)
target_link_libraries(NNLSTMMLLabeler ${LIBS})
