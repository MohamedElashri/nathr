string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef unique)
set(stage "${NATHR_BUILD_DIR}/install-test-${unique}")
set(prefix "${stage}/relocated package")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${NATHR_BUILD_DIR}" --config "${NATHR_CONFIG}"
    --prefix "${stage}/prefix" COMMAND_ERROR_IS_FATAL ANY)
file(RENAME "${stage}/prefix" "${prefix}")

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${NATHR_SOURCE_DIR}/tests/consumer"
    -B "${stage}/consumer" -G Ninja "-DCMAKE_CXX_COMPILER=${NATHR_CXX}"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_BUILD_TYPE=${NATHR_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${stage}/consumer" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${stage}/consumer/nathr-consumer" "${NATHR_SOURCE_DIR}/tests/fixtures/model"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${prefix}/${NATHR_BINDIR}/nathr" forward
    --model "${NATHR_SOURCE_DIR}/tests/fixtures/tiny.gguf" --tokens 1,2,3 COMMAND_ERROR_IS_FATAL ANY)
file(REMOVE_RECURSE "${stage}")
