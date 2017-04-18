
include_directories(
  model
)

add_executable(NNRNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNRNNLabeler ${LIBS})
