@echo off

if "%1" == "" (
    echo "Error! Input video path missing."
)
if "%2" == "" (
    echo "Error! Output video path missing."
)
if "%3" == "" (
    echo "Error! Debugging mode missing."
)
if "%4" == "" (
    echo "Error! Debugging value missing." 
)

if "%3" == "--debug" (
    if "%4" == "0" (
        conda activate base
        python lane_detection.py %1 %2 %4
    ) else (
        if "%4" == "1" (
            conda activate base
            python lane_detection.py %1 %2 %4
        ) else (
            echo "Error! Debugging value must be 0 or 1"
        )
    )
    
) else (
    echo "Error! --debug keyword missing."
)
pause