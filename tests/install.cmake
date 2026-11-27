string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef unique)
set(stage "${NATHR_BUILD_DIR}/install-test-${unique}")
set(releases "${stage}/relocated package/releases")
set(prefix "${releases}/nathr-${NATHR_VERSION}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${NATHR_BUILD_DIR}" --config "${NATHR_CONFIG}"
    --prefix "${stage}/prefix" COMMAND_ERROR_IS_FATAL ANY)
file(MAKE_DIRECTORY "${releases}")
file(RENAME "${stage}/prefix" "${prefix}")

execute_process(COMMAND "${CMAKE_COMMAND}" -S "${NATHR_SOURCE_DIR}/tests/consumer"
    -B "${stage}/consumer" -G Ninja "-DCMAKE_CXX_COMPILER=${NATHR_CXX}"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_BUILD_TYPE=${NATHR_CONFIG}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${stage}/consumer" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${stage}/consumer/nathr-consumer" "${NATHR_SOURCE_DIR}/tests/fixtures/model"
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${prefix}/${NATHR_BINDIR}/nathr" forward
    --model "${NATHR_SOURCE_DIR}/tests/fixtures/tiny.gguf" --tokens 1,2,3 COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${prefix}/${NATHR_BINDIR}/nathr-serve.py")
    message(FATAL_ERROR "Installed CLI is missing its HTTP frontend")
endif()
foreach(asset IN ITEMS index.html app.css app.js)
    if(NOT EXISTS "${prefix}/${NATHR_BINDIR}/nathr-web/${asset}")
        message(FATAL_ERROR "Installed CLI is missing browser asset ${asset}")
    endif()
endforeach()
set(model_data "${prefix}/${NATHR_DATADIR}/nathr")
execute_process(COMMAND "${model_data}/setup.sh" --help COMMAND_ERROR_IS_FATAL ANY)
if(NOT EXISTS "${model_data}/model_store.py")
    message(FATAL_ERROR "Installed CLI is missing its model downloader")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${model_data}/models/manifest.json"
    "${NATHR_SOURCE_DIR}/models/manifest.json" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${prefix}/${NATHR_DOCDIR}/LICENSE"
    "${NATHR_SOURCE_DIR}/LICENSE" COMMAND_ERROR_IS_FATAL ANY)
if(NATHR_PYTHON AND NATHR_BINDIR STREQUAL "bin")
    execute_process(COMMAND "${NATHR_PYTHON}" -B "${NATHR_SOURCE_DIR}/tests/update.py"
        "${prefix}" "${model_data}" "${stage}" "${NATHR_SOURCE_DIR}/tests/fixtures/t5"
        COMMAND_ERROR_IS_FATAL ANY)
endif()
file(REMOVE_RECURSE "${stage}")
