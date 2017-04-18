
include_directories(
  model
)

add_executable(NNBiLSTMLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiLSTMLabeler ${LIBS})
