
include_directories(
  model
)

add_executable(NNGRNNMLLabeler NNCNNLabeler.cpp)
target_link_libraries(NNGRNNMLLabeler ${LIBS})
