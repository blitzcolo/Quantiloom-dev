execute_process(COMMAND "${CLI}" --version
    RESULT_VARIABLE result OUTPUT_VARIABLE actual ERROR_VARIABLE error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT result EQUAL 0 OR NOT actual STREQUAL "Quantiloom ${EXPECTED}")
    message(FATAL_ERROR "CLI version mismatch: '${actual}', expected Quantiloom ${EXPECTED}. ${error}")
endif()
