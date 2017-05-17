
include_directories(
  model
)

add_executable(NNCNNMLLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNMLLabeler ${LIBS})
