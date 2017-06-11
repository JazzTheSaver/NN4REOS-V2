
include_directories(
  model
)

add_executable(NNBiGRNNMLCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNBiGRNNMLCharLabeler ${LIBS})
