
include_directories(
  model
)

add_executable(NNCNNCharLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNCharLabeler ${LIBS})
