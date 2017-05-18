
include_directories(
  model
)

add_executable(NNRNNMLLabeler NNCNNLabeler.cpp)
target_link_libraries(NNRNNMLLabeler ${LIBS})
