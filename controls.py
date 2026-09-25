"""
Controls page (/controls) - two columns:
  left:  Actuator, Gates (servos), Pump/Feeder (relays)
  right: Automation Timing

"Servo 1/2" are labeled "Gate 1/2" here (still driven by the same S1:/S2:
serial commands under the hood). Relay 3 has no manual control on this
page since nothing uses it; Relay 1 is labeled "Pump" (matches its use in
the automation preset) and Relay 2 is labeled "Feeder" (matches its use in
the feed-dispense loop).
"""

from flask import Blueprint, Response


def create_controls_blueprint(
    render_page,
    servo1_open_deg,
    servo1_close_deg,
    servo2_open_deg,
    servo2_close_deg,
):
    """render_page and the servo degree values come from main.py,
    passed in here rather than imported directly, so this module has
    no dependency on main.py (avoids a circular import).
    """

    CONTROLS_BODY = f"""
        <div class="row">
          <div class="col-lg-6">

            <div class="card card-outline card-warning">
              <div class="card-header">
                <h3 class="card-title">Actuator</h3>
              </div>
              <div class="card-body text-center">
                <p class="text-muted small">Hold UP or DOWN to move &mdash; release to stop</p>
                <div class="btn-group" role="group">
                  <button id="actUp" class="btn btn-success btn-lg">&#9650; UP</button>
                  <button id="actDown" class="btn btn-warning btn-lg">&#9660; DOWN</button>
                  <button id="actStop" class="btn btn-danger btn-lg">&#9632; STOP</button>
                </div>
              </div>
            </div>

            <div class="card card-outline card-info">
              <div class="card-header">
                <h3 class="card-title">Gates</h3>
              </div>
              <div class="card-body">
                <div class="d-flex align-items-center justify-content-between py-2 border-bottom">
                  <div>
                    <div class="font-weight-bold">Gate 1</div>
                    <small class="text-muted">Close={servo1_close_deg}&deg; &middot; Open={servo1_open_deg}&deg;</small>
                  </div>
                  <div class="custom-control custom-switch">
                    <input type="checkbox" class="custom-control-input" id="servo1Toggle">
                    <label class="custom-control-label" for="servo1Toggle"></label>
                  </div>
                </div>
                <div class="d-flex align-items-center justify-content-between py-2">
                  <div>
                    <div class="font-weight-bold">Gate 2</div>
                    <small class="text-muted">Close={servo2_close_deg}&deg; &middot; Open={servo2_open_deg}&deg;</small>
                  </div>
                  <div class="custom-control custom-switch">
                    <input type="checkbox" class="custom-control-input" id="servo2Toggle">
                    <label class="custom-control-label" for="servo2Toggle"></label>
                  </div>
                </div>
                
                  <div class="d-flex align-items-center justify-content-between py-2 border-bottom">
                  <span class="font-weight-bold">Pump</span>
                  <div class="custom-control custom-switch">
                    <input type="checkbox" class="custom-control-input" id="relay1Toggle" data-relay="1">
                    <label class="custom-control-label" for="relay1Toggle"></label>
                  </div>
              </div>    
              
               <div class="d-flex align-items-center justify-content-between py-2">
                  <span class="font-weight-bold">Feeder</span>
                  <div class="custom-control custom-switch">
                    <input type="checkbox" class="custom-control-input" id="relay2Toggle" data-relay="2">
                    <label class="custom-control-label" for="relay2Toggle"></label>
                  </div>
                </div>
              </div>
            </div>

            </div>


          <div class="col-lg-6">

            <div class="card card-outline card-primary">
              <div class="card-header">
                <h3 class="card-title">Automation Timing</h3>
              </div>
              <div class="card-body">
                <p class="text-muted small">
                  Adjust how long each step of the automation loop holds, plus how long
                  the feeder stays ON each dispense pulse. Saving these values makes them
                  the system default for later automation runs.
                </p>
                <div id="automationDurations"></div>
                <button id="applyDurationsBtn" class="btn btn-primary btn-block mt-2">Save Timing</button>
                <p id="durationsStatus" class="text-center text-muted mt-2 mb-0">&nbsp;</p>
              </div>
            </div>

          </div>
        </div>
"""

    CONTROLS_SCRIPT = f"""
const relayState = {{1:false, 2:false}};

// Servo config: open/close angles, and current state (false = closed, matches ESP32 default)
const servoConfig = {{
  1: {{ open: {servo1_open_deg}, close: {servo1_close_deg} }},
  2: {{ open: {servo2_open_deg}, close: {servo2_close_deg} }}
}};
const servoState = {{ 1: false, 2: false }};

async function sendCommand(cmd){{
  const res = await fetch('/api/send', {{
    method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{command: cmd}})
  }});
  const data = await res.json();
  if (!data.ok) alert('Send failed: ' + data.error);
}}

// ---- Actuator: press-and-hold (moves while held, stops on release) ----
function bindHoldButton(buttonId, startCommand){{
  const btn = document.getElementById(buttonId);
  const start = (e) => {{ e.preventDefault(); sendCommand(startCommand); }};
  const stop = () => sendCommand('AS');
  btn.addEventListener('mousedown', start);
  btn.addEventListener('mouseup', stop);
  btn.addEventListener('mouseleave', stop);
  btn.addEventListener('touchstart', start, {{passive:false}});
  btn.addEventListener('touchend', stop);
  btn.addEventListener('touchcancel', stop);
}}
bindHoldButton('actUp', 'AU:99');
bindHoldButton('actDown', 'AD:99');
document.getElementById('actStop').addEventListener('click', () => sendCommand('AS'));

// ---- Gate (servo) OPEN/CLOSE toggle switches ----
function bindServoToggle(inputId, num){{
  const input = document.getElementById(inputId);
  input.addEventListener('change', () => {{
    servoState[num] = input.checked;
    const deg = input.checked ? servoConfig[num].open : servoConfig[num].close;
    sendCommand('S' + num + ':' + deg);
  }});
}}
bindServoToggle('servo1Toggle', 1);
bindServoToggle('servo2Toggle', 2);

document.querySelectorAll('.custom-control-input[data-relay]').forEach(input => {{
  const num = input.dataset.relay;
  input.addEventListener('change', () => {{
    relayState[num] = input.checked;
    sendCommand('R' + num + (input.checked ? 'ON' : 'OFF'));
  }});
}});

// ---- Automation Timing (durations for the fixed preset sequence) ----
// DEVICE_LABELS is already declared globally by the shared navbar script -
// reusing it here (not redeclaring) is what keeps this page's connect
// button etc. working: a duplicate `const DEVICE_LABELS` would throw a
// SyntaxError and silently break every script on this page, including the
// serial Connect button's click handler.

const durationsContainer = document.getElementById('automationDurations');
const durationsStatusEl = document.getElementById('durationsStatus');

function appendDurationRow(label, value, inputClass, inputId){{
  const row = document.createElement('div');
  row.className = 'form-row align-items-center mb-2';

  const labelCol = document.createElement('div');
  labelCol.className = 'col-7';
  labelCol.textContent = label;

  const inputCol = document.createElement('div');
  inputCol.className = 'col-5';
  const group = document.createElement('div');
  group.className = 'input-group input-group-sm';
  const input = document.createElement('input');
  input.type = 'number';
  input.min = '0';
  input.step = '0.5';
  input.className = 'form-control ' + inputClass;
  if (inputId) input.id = inputId;
  input.value = value;
  const append = document.createElement('div');
  append.className = 'input-group-append';
  const appendText = document.createElement('span');
  appendText.className = 'input-group-text';
  appendText.textContent = 's';
  append.appendChild(appendText);
  group.appendChild(input);
  group.appendChild(append);
  inputCol.appendChild(group);

  row.appendChild(labelCol);
  row.appendChild(inputCol);
  durationsContainer.appendChild(row);
}}

async function loadAutomationDurations(){{
  try{{
    const res = await fetch('/api/automation');
    const data = await res.json();
    durationsContainer.innerHTML = '';
    data.steps.forEach((step) => {{
      appendDurationRow(
        (DEVICE_LABELS[step.device] || step.device) + ' \\u2192 ' + step.action.toUpperCase(),
        step.duration,
        'duration-input'
      );
    }});
    appendDurationRow(
      'Feeder \\u2192 ON',
      data.feeder_seconds != null ? data.feeder_seconds : 5,
      'feeder-duration-input',
      'feederDurationInput'
    );
  }} catch(e){{
    durationsContainer.innerHTML = '<p class="text-muted">Could not load timing.</p>';
  }}
}}

document.getElementById('applyDurationsBtn').addEventListener('click', async () => {{
  const durations = Array.from(
    durationsContainer.querySelectorAll('.duration-input')
  ).map(input => parseFloat(input.value) || 0);
  const feederInput = document.getElementById('feederDurationInput');
  const feeder_seconds = feederInput ? (parseFloat(feederInput.value) || 0) : undefined;
  try{{
    const res = await fetch('/api/automation/durations', {{
      method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify({{durations, feeder_seconds}})
    }});
    const data = await res.json();
    durationsStatusEl.textContent = data.ok ? '\\u2713 Timing saved as default' : (data.error || 'Could not save');
  }} catch(e){{
    durationsStatusEl.textContent = 'Could not save timing';
  }}
}});

loadAutomationDurations();
"""

    controls_bp = Blueprint("controls", __name__)

    @controls_bp.route("/controls")
    def controls_page():
        html = render_page("controls", CONTROLS_BODY, CONTROLS_SCRIPT)
        return Response(html, mimetype="text/html")

    return controls_bp
