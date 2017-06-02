
include_directories(
  model
)

add_executable(NNCNNMLCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNMLCharLabeler ${LIBS})
