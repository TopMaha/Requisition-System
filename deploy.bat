@echo off
echo =====================================================
echo   Requisition System - Deploy Worker to Cloudflare
echo =====================================================
echo.
echo  D1 Database : requisition-db  [Created]
echo  Database ID : eab82fe8-726a-46a1-87fd-8b90a9702e92
echo  Tables      : categories, items, requisitions,
echo                requisition_items, stock_movements  [Created]
echo.

where wrangler >nul 2>&1
if %errorlevel% neq 0 (
  echo [!] wrangler not found - installing...
  npm install -g wrangler
  echo.
)

echo [1/2] Checking Cloudflare login...
wrangler whoami >nul 2>&1
if %errorlevel% neq 0 (
  echo     Not logged in - opening browser...
  wrangler login
)
echo     Logged in OK
echo.

echo [2/2] Deploying Worker to Cloudflare...
wrangler deploy
if %errorlevel% neq 0 (
  echo.
  echo [!] Deploy failed - check errors above
  pause
  exit /b 1
)

echo.
echo =====================================================
echo   Deploy complete!
echo   Copy the Worker URL (*.workers.dev) above
echo   Open index.html - click Settings icon
echo   Paste the URL in "Worker URL" field
echo =====================================================
pause
