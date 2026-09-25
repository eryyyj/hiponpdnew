"""
Splash page (/splash) - a plain white full-screen page showing
assets/images/landing.png, shown by the kiosk browser on startup before
it redirects to the main interface ("/").

Deliberately does NOT use the shared render_page() shell (no topbar, no
nav) - a splash screen should be nothing but the image on a blank
background until the real interface takes over.
"""

from flask import Blueprint, Response


def create_splash_blueprint(duration_ms, bg_color="white"):
    """
    render_page isn't needed here since this page has no topbar/nav, but
    the factory-function pattern (matching camera.py/controls.py/
    gallery.py) still applies for the same reason: this module stays
    independent of main.py, avoiding a circular import.
    """

    splash_bp = Blueprint("splash", __name__)

    @splash_bp.route("/splash")
    def splash_page():
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>Shrimp Farm Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700;900&display=swap" rel="stylesheet">
<style>
html, body {{
  margin: 0;
  padding: 0;
  width: 100vw;
  height: 100vh;
  background: {bg_color};
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}

#splash-image {{
  max-width: 65vw;
  max-height: 65vh;
  width: auto;
  height: auto;
  object-fit: contain;
  transform: translateY(-10vh);
}}
  #splash-fallback {{
    display: none;
    font-family: 'Lato', sans-serif;
    font-size: 28px;
    font-weight: 900;
    color: #ff8a65;
  }}
</style>
</head>
<body>
  <img
    id="splash-image"
    src="/assets/images/landing.png"
    alt="Shrimp Farm Control"
    onerror="this.style.display='none'; document.getElementById('splash-fallback').style.display='block';"
  >
  <div id="splash-fallback">&#129424; Shrimp Farm Control</div>

  <script>
    setTimeout(() => {{ window.location.href = "/"; }}, {duration_ms});
  </script>
</body>
</html>
"""
        return Response(html, mimetype="text/html")

    return splash_bp
