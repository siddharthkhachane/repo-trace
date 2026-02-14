@echo off
setlocal

set BASE_URL=http://127.0.0.1:8000
set FILE_PATH=app/main.py

echo [1/3] History for %FILE_PATH%
curl -s "%BASE_URL%/api/time-travel/history?path=%FILE_PATH%"
echo.
echo.

echo [2/3] Snapshot (latest chapter hash inferred manually from step 1)
echo Example:
echo curl -s "%BASE_URL%/api/time-travel/snapshot?path=%FILE_PATH%^&commit=COMMIT_HASH"
echo.

echo [3/3] Diff (previous -> current chapter hash inferred manually from step 1)
echo Example:
echo curl -s "%BASE_URL%/api/time-travel/diff?path=%FILE_PATH%^&from=FROM_HASH^&to=TO_HASH"
echo.

endlocal
