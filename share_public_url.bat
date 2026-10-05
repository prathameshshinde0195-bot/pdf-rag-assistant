@echo off
echo =====================================================================
echo 🌐 INSTANT FREE PUBLIC DEPLOYMENT (Cloudflare Quick Tunnel)
echo =====================================================================
echo.
echo This will create a secure, public HTTPS link (e.g. https://xxxx.trycloudflare.com)
echo that anyone in the world can open on their phone or laptop to test your app!
echo.
echo Make sure your Streamlit app is running in another terminal (streamlit run app.py).
echo.
echo =====================================================================
echo Starting Cloudflare Tunnel on http://localhost:8501...
echo Look for the "https://....trycloudflare.com" link below!
echo =====================================================================
echo.

:: Run Cloudflare Tunnel binary or npx cloudflared
npx -y cloudflared tunnel --url http://localhost:8501

pause
