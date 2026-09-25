"""
Gallery page (/gallery) - grid of saved snapshots with a lightbox viewer.

The Start/Stop Loop control (for the fixed preset automation sequence)
lives in the shared top navbar (see main.py), not on this page, so it's
visible and stays in sync everywhere rather than being duplicated per page.
"""

from flask import Blueprint, Response

GALLERY_BODY = """
        <div class="card card-outline card-primary">
          <div class="card-header">
            <h3 class="card-title">Snapshots</h3>
            <div class="card-tools">
              <button id="galleryRefreshBtn" class="btn btn-sm btn-outline-secondary">&#8635; Refresh</button>
            </div>
          </div>
          <div class="card-body">
            <div id="galleryGrid" class="row"></div>
          </div>
        </div>
"""

GALLERY_EXTRA_BODY = """
<div id="lightbox" class="lightbox hidden">
  <span class="lightbox-close" id="lightboxClose">&times;</span>
  <img id="lightboxImg" src="" alt="">
  <div class="lightbox-caption" id="lightboxCaption"></div>
</div>
"""

GALLERY_SCRIPT = """
async function loadGallery(){
  const grid = document.getElementById('galleryGrid');
  try{
    const res = await fetch('/api/gallery');
    const data = await res.json();
    grid.innerHTML = '';
    if (!data.items.length){
      grid.innerHTML = '<div class="col-12 text-center text-muted py-5">No snapshots yet. The camera saves one automatically when a detection run finishes.</div>';
      return;
    }
    data.items.forEach(item => {
      const col = document.createElement('div');
      col.className = 'col-6 col-md-3 col-lg-2 mb-3 gallery-item';
      col.innerHTML = `<img src="${item.url}" alt="${item.filename}"><div class="caption">${item.mtime}</div>`;
      col.addEventListener('click', () => openLightbox(item.url, item.filename + ' \\u2022 ' + item.mtime));
      grid.appendChild(col);
    });
  } catch(e){
    grid.innerHTML = '<div class="col-12 text-center text-muted py-5">Could not load gallery.</div>';
  }
}

document.getElementById('galleryRefreshBtn').addEventListener('click', loadGallery);

function openLightbox(url, caption){
  document.getElementById('lightboxImg').src = url;
  document.getElementById('lightboxCaption').textContent = caption || '';
  document.getElementById('lightbox').classList.remove('hidden');
}

function closeLightbox(){
  document.getElementById('lightbox').classList.add('hidden');
}

document.getElementById('lightbox').addEventListener('click', closeLightbox);
document.getElementById('lightboxClose').addEventListener('click', closeLightbox);

loadGallery();
setInterval(() => {
  if (!document.getElementById('lightbox').classList.contains('hidden')) return;
  loadGallery();
}, 3000);
"""


def create_gallery_blueprint(render_page):
    """render_page comes from main.py, passed in here rather than
    imported directly, so this module has no dependency on main.py
    (avoids a circular import).
    """
    gallery_bp = Blueprint("gallery", __name__)

    @gallery_bp.route("/gallery")
    def gallery_page():
        html = render_page("gallery", GALLERY_BODY, GALLERY_SCRIPT, extra_body=GALLERY_EXTRA_BODY)
        return Response(html, mimetype="text/html")

    return gallery_bp
