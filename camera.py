"""
Camera page (/) - live IMX500 feed with shrimp detection overlay.

The page contains:
    - Live camera feed
    - LIVE indicator
    - Camera counting/tracking line
    - Start Loop button
    - Cancel Loop button
    - Flush button
    - Automation status below the buttons
    - Flush status below the automation status

The automation backend is shared with main.py. This page does NOT create
another automation manager. It only communicates with the existing API
endpoints:

    /api/automation
    /api/automation/start
    /api/automation/stop
    /api/shrimp_count

Flush uses:

    /api/flush/start
    /api/flush/status

The shared render_page() function is passed into this blueprint from main.py
to avoid a circular import.
"""

from flask import Blueprint, Response


# ============================================================================
# CAMERA PAGE HTML
# ============================================================================

CAMERA_BODY = """
        <div class="card">
          <div class="card-body camera-split">

            <div class="camera-split-left">
              <div class="camera-frame" id="cameraFrame">
                <div class="camera-stage" id="cameraStage">
                  <img
                    id="videoFeed"
                    src="/video_feed"
                    alt="Live camera feed"
                  >
                  <div class="count-line" id="countLine">
                    <div class="count-line-bar"></div>
                  </div>
                </div>
                <div class="live-badge">
                  <span class="live-dot"></span>
                  LIVE
                </div>
              </div>

              <div class="d-flex justify-content-center flex-wrap mt-1 camera-split-actions" style="gap:6px;">
                <button id="cameraStartLoopBtn" class="btn btn-success feeder-action-btn">&#9654; Start Loop</button>
                <button id="cameraCancelLoopBtn" class="btn btn-danger feeder-action-btn">&#10005; Cancel</button>
                <button id="cameraFlushBtn" class="btn btn-info feeder-action-btn">Flush</button>
              </div>
              <div id="cameraAutomationStatus" class="text-center mt-1 mb-0" style="min-height:18px; font-size:12px;">
                <span class="text-muted">Automation: Idle</span>
              </div>
              <p id="cameraFlushStatus" class="text-center text-muted mt-0 mb-0" style="font-size:12px;">&nbsp;</p>
            </div>

            <div class="camera-split-right" id="feedDetailsPanel">
              <div class="feed-side-title">Feed</div>
              <label for="feederCountInput" class="feed-side-label" id="feederCountLabel">To Count</label>
              <input id="feederCountInput" type="text" inputmode="none" autocomplete="off" class="form-control" placeholder="0">
              <p class="feeder-stat" id="feederTargetCountWrap" style="display:none;">Target Count<br><strong id="feederTargetCountValue">0</strong></p>
              <p class="feeder-stat">Total Biomass<br><strong id="feederBiomassValue">0.00 g</strong></p>
              <p class="feeder-stat">Feed to dispense<br><strong id="feederDispenseValue">0.00 g</strong></p>
              <p class="feeder-stat">Total current weight<br><strong id="feederCurrentWeightValue">0.00 g</strong></p>
              <button type="button" id="feederStartBtn" class="btn btn-warning btn-block feeder-action-btn">Start</button>
              <button type="button" id="feederStopBtn" class="btn btn-outline-danger btn-block feeder-action-btn mb-2">Stop</button>
              <button type="button" id="feederManualBtn" class="btn btn-secondary btn-block feeder-action-btn">Manual Feeder</button>
              <p id="feederLiveStatus" class="text-muted mt-2 mb-0" style="font-size:16px; min-height:20px;"></p>
            </div>

          </div>
        </div>
"""


# Toast container retained for compatibility with the shared page.
CAMERA_EXTRA_BODY = """
<div
  id="captureToast"
  class="alert alert-success shadow"
></div>
<div id="feederModal">
  <div class="modal-dialog modal-dialog-centered">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Manual Feeder</h5>
        <button type="button" class="close" id="feederModalClose" aria-label="Close"><span aria-hidden="true">&times;</span></button>
      </div>
      <div class="modal-body">
        <div class="d-flex justify-content-center flex-wrap" style="gap:10px;">
          <button type="button" id="feederManualOnBtn" class="btn btn-success feeder-action-btn">ON</button>
          <button type="button" id="feederManualOffBtn" class="btn btn-outline-secondary feeder-action-btn">OFF</button>
          <button type="button" id="feederResetBtn" class="btn btn-outline-danger feeder-action-btn">Reset</button>
        </div>
      </div>
    </div>
  </div>
</div>
"""


# ============================================================================
# CAMERA PAGE JAVASCRIPT
# ============================================================================

CAMERA_SCRIPT = r"""
// ============================================================================
// CAMERA PAGE AUTOMATION
// ============================================================================
//
// This page uses the SAME backend automation manager as main.py.
//
// It does not create another automation manager.
//
// Shared functions supplied by main.py include:
//
//     openShrimpTargetModal()
//     openFeedReadyModal()
//     DEVICE_LABELS
//
// The camera page itself owns:
//
//     Start Loop button
//     Cancel Loop button
//     Flush button
//     Automation status display
//     Flush status display
//
// ============================================================================


// ============================================================================
// ELEMENTS
// ============================================================================

const cameraStartLoopBtn =
  document.getElementById('cameraStartLoopBtn');

const cameraCancelLoopBtn =
  document.getElementById('cameraCancelLoopBtn');

const cameraFlushBtn =
  document.getElementById('cameraFlushBtn');

const cameraAutomationStatusEl =
  document.getElementById('cameraAutomationStatus');

const cameraFlushStatusEl =
  document.getElementById('cameraFlushStatus');


// ============================================================================
// LOCAL STATE
// ============================================================================

let flushRunning = false;


// ============================================================================
// AUTOMATION STATUS
// ============================================================================

function renderCameraAutomationStatus(data, currentCount) {

  // Keep the shared automation state synchronized.
  automationRunning = !!data.running;

  const steps = data.steps || [];

  const step =
    steps[data.current_index];

  // ------------------------------------------------------------
  // RUNNING
  // ------------------------------------------------------------

  if (automationRunning && step) {

    const deviceLabel =
      (typeof DEVICE_LABELS !== 'undefined' &&
      DEVICE_LABELS[step.device])
        ? DEVICE_LABELS[step.device]
        : step.device;

    const action =
      step.action
        ? step.action.toUpperCase()
        : '';

    const label =
      deviceLabel +
      ' \u2192 ' +
      action;

    const tail =
      data.seconds_left
        ? ' \u2014 ' + data.seconds_left + 's'
        : '';

    let progress = '';

    if (
      data.target_count != null &&
      currentCount != null
    ) {

      progress =
        ' (' +
        currentCount +
        '/' +
        data.target_count +
        ' shrimp)';

    }

    cameraAutomationStatusEl.innerHTML =
      '<strong>Automation:</strong> ' +
      label +
      tail +
      progress;

  }

  // ------------------------------------------------------------
  // IDLE
  // ------------------------------------------------------------

  else {

    cameraAutomationStatusEl.innerHTML =
      '<span class="text-muted">' +
      'Automation: Idle' +
      '</span>';

  }


  updateCameraButtonStates();
}


// ============================================================================
// POLL AUTOMATION STATUS
// ============================================================================

async function pollCameraAutomationStatus() {

  try {

    const results =
      await Promise.all([
        fetch('/api/automation'),
        fetch('/api/shrimp_count')
      ]);

    const autoRes =
      results[0];

    const countRes =
      results[1];


    if (!autoRes.ok) {
      throw new Error(
        'Automation API returned HTTP ' +
        autoRes.status
      );
    }


    if (!countRes.ok) {
      throw new Error(
        'Shrimp count API returned HTTP ' +
        countRes.status
      );
    }


    const autoData =
      await autoRes.json();

    const countData =
      await countRes.json();


    renderCameraAutomationStatus(
      autoData,
      countData.count
    );

    if (typeof syncFeederAutoTarget === 'function') {
      syncFeederAutoTarget(autoData);
    }


    // ----------------------------------------------------------
    // Automation completed
    // ----------------------------------------------------------

    if (
      typeof maybeOpenFeedReadyModal === 'function'
    ) {
      maybeOpenFeedReadyModal(autoData);
    } else if (
      typeof openFeedReadyModal === 'function' &&
      (autoData.just_completed || autoData.feed_popup_pending) &&
      autoData.completed_count != null
    ) {
      openFeedReadyModal(
        autoData.completed_count,
        autoData.completed_feed_grams,
        autoData.completed_target
      );
    }

  }

  catch (error) {

    // Do not spam the screen with temporary network errors.
    // The next polling cycle will automatically retry.
    console.warn(
      '[Camera] Automation status:',
      error
    );

  }

}


// ============================================================================
// START LOOP
// ============================================================================

cameraStartLoopBtn.addEventListener(
  'click',
  () => {

    if (
      automationRunning ||
      flushRunning
    ) {
      return;
    }


    if (
      typeof openShrimpTargetModal === 'function'
    ) {

      openShrimpTargetModal();

    }

  }
);


// ============================================================================
// CANCEL LOOP
// ============================================================================

cameraCancelLoopBtn.addEventListener(
  'click',
  async () => {

    cameraCancelLoopBtn.disabled = true;

    try {

      const res =
        await fetch(
          '/api/automation/stop',
          {
            method: 'POST'
          }
        );


      const data =
        await res.json();


      if (!data.ok) {

        console.warn(
          '[Camera] Could not stop automation:',
          data.error
        );

      }

    }

    catch (error) {

      console.warn(
        '[Camera] Stop request failed:',
        error
      );

    }


    // Immediately refresh the displayed state.
    await pollCameraAutomationStatus();

  }
);


// ============================================================================
// FLUSH START
// ============================================================================

cameraFlushBtn.addEventListener(
  'click',
  async () => {

    if (
      automationRunning ||
      flushRunning
    ) {
      return;
    }


    cameraFlushBtn.disabled = true;


    try {

      const res =
        await fetch(
          '/api/flush/start',
          {
            method: 'POST'
          }
        );


      const data =
        await res.json();


      if (!data.ok) {

        cameraFlushStatusEl.textContent =
          data.error ||
          'Could not start flush';

        updateCameraButtonStates();

        return;
      }


      cameraFlushStatusEl.innerHTML =
        '<strong>Flushing...</strong>';

    }

    catch (error) {

      console.warn(
        '[Camera] Flush request failed:',
        error
      );

      cameraFlushStatusEl.textContent =
        'Could not start flush';

    }


    await pollFlushStatus();

  }
);


// ============================================================================
// RENDER FLUSH STATUS
// ============================================================================

function renderFlushStatus(status) {

  flushRunning =
    !!status.running;


  const steps =
    status.steps || [];


  const step =
    steps[status.current_index];


  // ------------------------------------------------------------
  // FLUSH RUNNING
  // ------------------------------------------------------------

  if (
    flushRunning &&
    step
  ) {

    const deviceLabel =
      (typeof DEVICE_LABELS !== 'undefined' &&
      DEVICE_LABELS[step.device])
        ? DEVICE_LABELS[step.device]
        : step.device;


    const action =
      step.action
        ? step.action.toUpperCase()
        : '';


    const label =
      deviceLabel +
      ' \u2192 ' +
      action;


    const tail =
      status.seconds_left
        ? ' \u2014 ' +
          status.seconds_left +
          's'
        : '';


    cameraFlushStatusEl.innerHTML =
      '<strong>Flushing:</strong> ' +
      label +
      tail;

  }


  // ------------------------------------------------------------
  // FLUSH IDLE
  // ------------------------------------------------------------

  else {

    cameraFlushStatusEl.innerHTML =
      '&nbsp;';

  }


  updateCameraButtonStates();

}


// ============================================================================
// POLL FLUSH STATUS
// ============================================================================

async function pollFlushStatus() {

  try {

    const res =
      await fetch(
        '/api/flush/status'
      );


    if (!res.ok) {
      throw new Error(
        'Flush API returned HTTP ' +
        res.status
      );
    }


    const status =
      await res.json();


    renderFlushStatus(status);

  }

  catch (error) {

    console.warn(
      '[Camera] Flush status:',
      error
    );

  }

}


// ============================================================================
// BUTTON STATES
// ============================================================================

function updateCameraButtonStates() {

  // ------------------------------------------------------------
  // START LOOP
  // ------------------------------------------------------------

  cameraStartLoopBtn.disabled =
    automationRunning ||
    flushRunning;


  // ------------------------------------------------------------
  // CANCEL LOOP
  // ------------------------------------------------------------

  cameraCancelLoopBtn.disabled =
    !automationRunning;


  // ------------------------------------------------------------
  // FLUSH
  // ------------------------------------------------------------

  cameraFlushBtn.disabled =
    automationRunning ||
    flushRunning;

}


// ============================================================================
// INITIAL STATE
// ============================================================================

updateCameraButtonStates();


// Get the initial automation state immediately.
pollCameraAutomationStatus();


// Get the initial flush state immediately.
pollFlushStatus();


// ============================================================================
// CONTINUOUS STATUS POLLING
// ============================================================================

setInterval(
  pollCameraAutomationStatus,
  1000
);


setInterval(
  pollFlushStatus,
  1000
);


setInterval(
  updateCameraButtonStates,
  500
);


// ============================================================================
// DRAGGABLE COUNTING LINE
// ============================================================================

const countLineEl = document.getElementById('countLine');
const cameraStageEl = document.getElementById('cameraStage');
let countLineFraction = 0.80;
let draggingCountLine = false;
let lastCountLineSave = 0;

function applyCountLineFraction(fraction){
  countLineFraction = Math.min(1.0, Math.max(0.08, fraction));
  countLineEl.style.top = (countLineFraction * 100) + '%';
}

async function loadCountLine(){
  try{
    const res = await fetch('/api/count_line');
    const data = await res.json();
    if (data.y_fraction != null) applyCountLineFraction(data.y_fraction);
  } catch(e){
    applyCountLineFraction(countLineFraction);
  }
}

async function saveCountLine(fraction){
  try{
    await fetch('/api/count_line', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({y_fraction: fraction})
    });
  } catch(e){ /* keep local position even if save fails */ }
}

function maybeSaveCountLine(force){
  const now = Date.now();
  if (!force && now - lastCountLineSave < 120) return;
  lastCountLineSave = now;
  saveCountLine(countLineFraction);
}

function fractionFromPointer(clientY){
  const rect = cameraStageEl.getBoundingClientRect();
  if (!rect.height) return countLineFraction;
  return (clientY - rect.top) / rect.height;
}

function startCountLineDrag(e){
  e.preventDefault();
  draggingCountLine = true;
  countLineEl.setPointerCapture(e.pointerId);
  applyCountLineFraction(fractionFromPointer(e.clientY));
}

countLineEl.addEventListener('pointerdown', startCountLineDrag);

countLineEl.addEventListener('pointermove', (e) => {
  if (!draggingCountLine) return;
  e.preventDefault();
  applyCountLineFraction(fractionFromPointer(e.clientY));
  maybeSaveCountLine(false);
});

function endCountLineDrag(e){
  if (!draggingCountLine) return;
  draggingCountLine = false;
  try{ countLineEl.releasePointerCapture(e.pointerId); } catch(err){}
  maybeSaveCountLine(true);
}

countLineEl.addEventListener('pointerup', endCountLineDrag);
countLineEl.addEventListener('pointercancel', endCountLineDrag);

loadCountLine();
"""


# ============================================================================
# BLUEPRINT FACTORY
# ============================================================================

def create_camera_blueprint(render_page):
    """
    Create the Camera Flask blueprint.

    render_page is supplied by main.py rather than imported directly.
    This keeps camera.py independent from main.py and prevents circular
    imports.
    """

    camera_bp = Blueprint(
        "camera",
        __name__
    )


    @camera_bp.route("/")
    def index():

        html = render_page(
            "camera",
            CAMERA_BODY,
            CAMERA_SCRIPT,
            extra_body=CAMERA_EXTRA_BODY,
            full_height=True
        )

        return Response(
            html,
            mimetype="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
            },
        )


    return camera_bp
