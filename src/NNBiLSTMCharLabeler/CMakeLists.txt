
include_directories(
  model
)

add_executable(NNBiLSTMCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiLSTMCharLabeler ${LIBS})
