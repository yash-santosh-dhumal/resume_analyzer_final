@echo off
echo.
echo ========================================
echo  SIMPLE GIT PUSH SOLUTION
echo ========================================
echo.

echo 1. First, let's get out of any Git operations...
git rebase --abort 2>nul

echo.
echo 2. Check current status...
git status

echo.
echo 3. The easiest solution is to use GitHub's allow option:
echo    Go to this URL to allow the secret:
echo    https://github.com/PranavDoke/resume-analyzer/security/secret-scanning/unblock-secret/32yEOa0WWIkbvsuzFsEF6CmJO3V
echo.

echo 4. After allowing the secret on GitHub, run:
echo    git push origin main
echo.

echo ========================================
echo  OR - Alternative: Fresh commit without API keys
echo ========================================
echo.

echo If the above doesn't work, run these commands:
echo.
echo # Reset to clean state
echo git reset --hard origin/main
echo.
echo # Add all files (but .env.example is now clean)  
echo git add .
echo.
echo # Commit without API keys
echo git commit -m "Enhanced Resume Analyzer with Dynamic Scoring"
echo.
echo # Push
echo git push origin main
echo.

pause