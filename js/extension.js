import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── COCO-17 skeleton ───────────────────────────────────────────────────────────
const COCO_JOINT_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
];

const COCO_LIMBS = [
    [0,1],[0,2],[1,3],[2,4],
    [5,7],[7,9],[6,8],[8,10],
    [5,6],[5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],
];

// Limb-group colors matching Hina reference: left=blue, right=red, center=green
// Order matches COCO_LIMBS: face(2), ears(2), L-arm(2), R-arm(2), shoulders(1), torso(2), hips(1), L-leg(2), R-leg(2)
const LIMB_COLORS = [
    "#44cc88", "#44cc88",    // nose→eyes (face, center)
    "#4d88ff", "#ff4d4d",    // left_eye→left_ear (left), right_eye→right_ear (right)
    "#4d88ff", "#4d88ff",    // left arm: shoulder→elbow→wrist
    "#ff4d4d", "#ff4d4d",    // right arm: shoulder→elbow→wrist
    "#44cc88",               // shoulder bar (center)
    "#44cc88", "#44cc88",    // torso sides (center)
    "#44cc88",               // hip bar (center)
    "#4d88ff", "#4d88ff",    // left leg: hip→knee→ankle
    "#ff4d4d", "#ff4d4d",    // right leg: hip→knee→ankle
];

// Per-joint group color (left=blue, right=red, center=green)
const COCO_JOINT_COLORS = [
    "#44cc88",               // 0  nose
    "#4d88ff", "#ff4d4d",    // 1  left_eye,       2  right_eye
    "#4d88ff", "#ff4d4d",    // 3  left_ear,        4  right_ear
    "#4d88ff", "#ff4d4d",    // 5  left_shoulder,   6  right_shoulder
    "#4d88ff", "#ff4d4d",    // 7  left_elbow,      8  right_elbow
    "#4d88ff", "#ff4d4d",    // 9  left_wrist,      10 right_wrist
    "#4d88ff", "#ff4d4d",    // 11 left_hip,        12 right_hip
    "#4d88ff", "#ff4d4d",    // 13 left_knee,       14 right_knee
    "#4d88ff", "#ff4d4d",    // 15 left_ankle,      16 right_ankle
];

const JOINT_RADIUS = 7;
const HIT_RADIUS   = 14;   // in world-space pixels (before zoom)
const H36M_JOINT_NAMES = [
    "sacrum",
    "left_hip", "left_knee", "left_foot",
    "right_hip", "right_knee", "right_foot",
    "center_torso", "upper_torso", "neck_base", "center_head",
    "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
];
const H36M_LIMBS = [
    [0, 1], [1, 2], [2, 3],
    [0, 4], [4, 5], [5, 6],
    [0, 7], [7, 8], [8, 9], [9, 10],
    [8, 11], [11, 12], [12, 13],
    [8, 14], [14, 15], [15, 16],
];

// ── Modal editor ───────────────────────────────────────────────────────────────

function openPoseEditorModal(kpsData, imageUrl, onApply) {
    // ── DOM skeleton ───────────────────────────────────────────────────────────
    const overlay = document.createElement("div");
    Object.assign(overlay.style, {
        position: "fixed", inset: "0",
        background: "rgba(0,0,0,0.75)",
        zIndex: "99999",
        display: "flex", alignItems: "center", justifyContent: "center",
    });

    const dialog = document.createElement("div");
    Object.assign(dialog.style, {
        background: "#1e1e2e", border: "1px solid #444", borderRadius: "8px",
        padding: "16px", display: "flex", flexDirection: "column", gap: "10px",
        maxWidth: "92vw", maxHeight: "92vh",
        boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
    });

    // Title row
    const title = document.createElement("div");
    title.textContent = "ClickPose Editor — drag joints to correct, then Apply";
    Object.assign(title.style, { color: "#ccc", fontSize: "13px", fontFamily: "sans-serif" });

    // Hint + controls row
    const controlRow = document.createElement("div");
    Object.assign(controlRow.style, {
        display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap",
    });

    const hint = document.createElement("div");
    hint.textContent = "● left  ● right  ● center  ● corrected  |  scroll=zoom  drag-bg=pan";
    Object.assign(hint.style, { color: "#777", fontSize: "11px", fontFamily: "sans-serif", flexShrink: "0" });

    // Opacity slider
    const opacityLabel = document.createElement("span");
    Object.assign(opacityLabel.style, { color: "#888", fontSize: "11px", fontFamily: "sans-serif" });
    opacityLabel.textContent = "Image:";

    const opacitySlider = document.createElement("input");
    opacitySlider.type = "range"; opacitySlider.min = "0"; opacitySlider.max = "100"; opacitySlider.value = "100";
    Object.assign(opacitySlider.style, { width: "90px", accentColor: "#4ec9b0", cursor: "pointer" });

    const opacityVal = document.createElement("span");
    Object.assign(opacityVal.style, { color: "#888", fontSize: "11px", fontFamily: "sans-serif", minWidth: "30px" });
    opacityVal.textContent = "100%";

    opacitySlider.addEventListener("input", () => {
        state.opacity = opacitySlider.value / 100;
        opacityVal.textContent = opacitySlider.value + "%";
        draw();
    });

    controlRow.append(hint, opacityLabel, opacitySlider, opacityVal);

    // Canvas
    const canvas = document.createElement("canvas");
    Object.assign(canvas.style, {
        display: "block", cursor: "crosshair",
        maxWidth: "80vw", maxHeight: "72vh",
        borderRadius: "4px", background: "#111",
    });

    // Button row
    const btnRow = document.createElement("div");
    Object.assign(btnRow.style, { display: "flex", gap: "8px", justifyContent: "flex-end" });

    const mkBtn = (label, bg) => {
        const b = document.createElement("button");
        b.textContent = label;
        Object.assign(b.style, {
            padding: "6px 18px", border: "none", borderRadius: "4px",
            background: bg, color: "#fff", cursor: "pointer",
            fontFamily: "sans-serif", fontSize: "13px",
        });
        return b;
    };
    const btnFit    = mkBtn("Fit",    "#444");
    const btnReset  = mkBtn("Reset",  "#555");
    const btnCancel = mkBtn("Cancel", "#666");
    const btnApply  = mkBtn("Apply",  "#2a7a4e");
    btnRow.append(btnFit, btnReset, btnCancel, btnApply);

    // Fixed joints info panel
    const infoPanel = document.createElement("div");
    Object.assign(infoPanel.style, {
        minHeight: "18px", fontFamily: "monospace", fontSize: "11px",
        color: "#4499ff", display: "flex", flexWrap: "wrap", gap: "8px",
    });

    dialog.append(title, controlRow, canvas, infoPanel, btnRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // ── state ──────────────────────────────────────────────────────────────────
    const state = {
        kps:      kpsData.keypoints.map(p => [p[0], p[1]]),
        scores:   [...kpsData.scores],
        origKps:  kpsData.keypoints.map(p => [p[0], p[1]]),
        imgW:     kpsData.image_size[1],
        imgH:     kpsData.image_size[0],
        baseScale: 1,      // initial fit-to-viewport scale (image → canvas world coords)
        img:      null,
        opacity:  1.0,
        // zoom / pan (applied on top of baseScale via ctx.transform)
        zoom:     1.0,
        panX:     0.0,
        panY:     0.0,
        // interaction
        dragging: null,    // joint index being dragged
        panning:  false,   // dragging the background
        panAnchorX: 0,
        panAnchorY: 0,
        moved:    new Set(),
    };

    // ── coordinate helpers ─────────────────────────────────────────────────────
    // Screen pixel (from mouse event) → canvas pixel (accounting for CSS scaling)
    function toScreen(e) {
        const r = canvas.getBoundingClientRect();
        return [
            (e.clientX - r.left) * (canvas.width  / r.width),
            (e.clientY - r.top)  * (canvas.height / r.height),
        ];
    }

    // Canvas screen pixel → world pixel (undoes zoom+pan transform)
    function screenToWorld(sx, sy) {
        return [
            (sx - state.panX) / state.zoom,
            (sy - state.panY) / state.zoom,
        ];
    }

    // World pixel → image pixel (undoes baseScale)
    function worldToImage(wx, wy) {
        return [wx / state.baseScale, wy / state.baseScale];
    }

    // Image pixel → world pixel
    function imageToWorld(ix, iy) {
        return [ix * state.baseScale, iy * state.baseScale];
    }

    // ── draw ───────────────────────────────────────────────────────────────────
    function draw() {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.save();
        ctx.translate(state.panX, state.panY);
        ctx.scale(state.zoom, state.zoom);

        // Background image
        if (state.img) {
            ctx.globalAlpha = state.opacity;
            ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
            ctx.globalAlpha = 1.0;
        }

        // Limbs
        for (let li = 0; li < COCO_LIMBS.length; li++) {
            const [i, j] = COCO_LIMBS[li];
            if (state.scores[i] > 0.05 && state.scores[j] > 0.05) {
                const [wx1, wy1] = imageToWorld(state.kps[i][0], state.kps[i][1]);
                const [wx2, wy2] = imageToWorld(state.kps[j][0], state.kps[j][1]);
                ctx.beginPath();
                ctx.strokeStyle = LIMB_COLORS[li] + "cc";
                ctx.lineWidth   = 3 / state.zoom;
                ctx.lineCap     = "round";
                ctx.moveTo(wx1, wy1);
                ctx.lineTo(wx2, wy2);
                ctx.stroke();
            }
        }

        // Joints
        for (let i = 0; i < state.kps.length; i++) {
            const [wx, wy] = imageToWorld(state.kps[i][0], state.kps[i][1]);
            const r = ((i === state.dragging) ? JOINT_RADIUS + 3 : JOINT_RADIUS) / state.zoom;

            const groupColor = COCO_JOINT_COLORS[i];
            const fill = (i === state.dragging) ? "#ffffff"
                       : state.moved.has(i)     ? "#ffe066"
                       : (state.scores[i] > 0.5) ? groupColor
                       :                           groupColor + "77";

            ctx.beginPath();
            ctx.arc(wx, wy, r, 0, Math.PI * 2);
            ctx.fillStyle   = fill;
            ctx.fill();
            ctx.strokeStyle = "rgba(0,0,0,0.6)";
            ctx.lineWidth   = 1.5 / state.zoom;
            ctx.stroke();

            // Label — shrinks with zoom so it stays readable
            const fs = Math.round(Math.max(8, Math.min(14, 10 / state.zoom)));
            ctx.fillStyle   = "#fff";
            ctx.font        = `bold ${fs}px sans-serif`;
            ctx.shadowColor = "black";
            ctx.shadowBlur  = 3;
            ctx.fillText(i + " " + COCO_JOINT_NAMES[i], wx + r + 2, wy + 4 / state.zoom);
            ctx.shadowBlur  = 0;
        }

        ctx.restore();

        // Update fixed joints info panel
        if (state.moved.size === 0) {
            infoPanel.textContent = "";
        } else {
            infoPanel.innerHTML = [...state.moved]
                .sort((a, b) => a - b)
                .map(i => {
                    const x = Math.round(state.kps[i][0]);
                    const y = Math.round(state.kps[i][1]);
                    return `<span>${i}&nbsp;${COCO_JOINT_NAMES[i]}&nbsp;[${x},${y}]</span>`;
                })
                .join("  ");
        }
    }

    // ── hit test (in screen coords) ────────────────────────────────────────────
    function hitTest(sx, sy) {
        const [wx, wy] = screenToWorld(sx, sy);
        let best = null, bestD = HIT_RADIUS;   // HIT_RADIUS is in world space
        for (let i = 0; i < state.kps.length; i++) {
            const [jwx, jwy] = imageToWorld(state.kps[i][0], state.kps[i][1]);
            const d = Math.hypot(wx - jwx, wy - jwy);
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    // ── mouse events ───────────────────────────────────────────────────────────
    canvas.addEventListener("mousedown", (e) => {
        e.preventDefault();
        const [sx, sy] = toScreen(e);
        const idx = hitTest(sx, sy);
        if (idx !== null) {
            state.dragging = idx;
            canvas.style.cursor = "grabbing";
        } else {
            // Pan on empty background
            state.panning    = true;
            state.panAnchorX = sx - state.panX;
            state.panAnchorY = sy - state.panY;
            canvas.style.cursor = "move";
        }
    });

    canvas.addEventListener("mousemove", (e) => {
        const [sx, sy] = toScreen(e);
        if (state.dragging !== null) {
            const [wx, wy] = screenToWorld(sx, sy);
            const [ix, iy] = worldToImage(wx, wy);
            state.kps[state.dragging][0] = Math.max(0, Math.min(state.imgW, ix));
            state.kps[state.dragging][1] = Math.max(0, Math.min(state.imgH, iy));
            state.scores[state.dragging] = 1.0;
            draw();
        } else if (state.panning) {
            state.panX = sx - state.panAnchorX;
            state.panY = sy - state.panAnchorY;
            draw();
        } else {
            canvas.style.cursor = hitTest(sx, sy) !== null ? "grab" : "crosshair";
        }
    });

    canvas.addEventListener("mouseup", () => {
        if (state.dragging !== null) state.moved.add(state.dragging);
        state.dragging = null;
        state.panning  = false;
        canvas.style.cursor = "crosshair";
        draw();
    });

    // Zoom toward mouse cursor
    canvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const [sx, sy] = toScreen(e);
        const factor   = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newZoom  = Math.max(0.25, Math.min(16, state.zoom * factor));
        // Keep the world point under the cursor fixed
        state.panX = sx - (sx - state.panX) * (newZoom / state.zoom);
        state.panY = sy - (sy - state.panY) * (newZoom / state.zoom);
        state.zoom = newZoom;
        draw();
    }, { passive: false });

    overlay.addEventListener("mousedown", (e) => {
        if (e.target === overlay) close();
    });

    // ── buttons ────────────────────────────────────────────────────────────────
    function close() { document.body.removeChild(overlay); }

    btnFit.addEventListener("click", () => {
        state.zoom = 1.0;
        state.panX = 0.0;
        state.panY = 0.0;
        draw();
    });

    btnCancel.addEventListener("click", close);

    btnReset.addEventListener("click", () => {
        state.kps    = state.origKps.map(p => [p[0], p[1]]);
        state.scores = [...kpsData.scores];
        state.moved.clear();
        draw();
    });

    btnApply.addEventListener("click", () => {
        // Only send joints the user actually moved
        const corrections = {};
        for (const i of state.moved) {
            corrections[String(i)] = [
                Math.round(state.kps[i][0]),
                Math.round(state.kps[i][1]),
            ];
        }
        onApply(corrections);
        close();
    });

    // ── load image + fit to viewport ───────────────────────────────────────────
    const img = new Image();
    img.onload = () => {
        const maxW = Math.min(window.innerWidth  * 0.80, 1280);
        const maxH = Math.min(window.innerHeight * 0.72, 900);
        const s    = Math.min(1, maxW / img.width, maxH / img.height);

        canvas.width      = Math.round(img.width  * s);
        canvas.height     = Math.round(img.height * s);
        state.baseScale   = s;
        state.img         = img;
        draw();
    };
    img.src = imageUrl;
}

function imageUrl(info) {
    return api.apiURL(
        `/view?filename=${encodeURIComponent(info.filename)}` +
        `&type=${encodeURIComponent(info.type)}` +
        `&subfolder=${encodeURIComponent(info.subfolder ?? "")}`
    );
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function detectExtensionFolder() {
    try {
        if (typeof import.meta !== "undefined" && import.meta.url) {
            const match = import.meta.url.match(/\/extensions\/([^/]+)\//);
            if (match) return match[1];
        }

        const scripts = document.getElementsByTagName("script");
        for (let i = scripts.length - 1; i >= 0; i--) {
            const src = scripts[i].src;
            if (!src) continue;
            const match = src.match(/\/extensions\/([^/]+)\//);
            if (match) return match[1];
        }
    } catch (_) {
        // Best effort only.
    }
    return null;
}

function createEmbeddedNativePose3DEditor(node) {
    const VIEWER_HEIGHT = 600;
    const NODE_MIN_WIDTH = 400;
    const NODE_CHROME_HEIGHT = 140;
    const NODE_MIN_HEIGHT = VIEWER_HEIGHT + NODE_CHROME_HEIGHT;

    const extensionFolder = detectExtensionFolder();
    const viewerUrl = extensionFolder
        ? `/extensions/${extensionFolder}/viewer_pose3d.html?v=${Date.now()}`
        : new URL("./viewer_pose3d.html", import.meta.url).href + `?v=${Date.now()}`;

    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "100%",
        height: `${VIEWER_HEIGHT}px`,
        minHeight: `${VIEWER_HEIGHT}px`,
        maxHeight: `${VIEWER_HEIGHT}px`,
        flex: "0 0 auto",
        position: "relative",
        overflow: "hidden",
        boxSizing: "border-box",
        borderRadius: "0",
        background: "#1a1a1a",
        border: "1px solid #333",
    });

    const iframe = document.createElement("iframe");
    iframe.src = viewerUrl;
    Object.assign(iframe.style, {
        width: "100%",
        height: "100%",
        display: "block",
        border: "none",
        background: "#1a1a1a",
    });

    container.append(iframe);

    let currentPose = null;
    let viewerReady = false;

    const postPose = () => {
        if (!viewerReady || !currentPose || !iframe.contentWindow) return;
        iframe.contentWindow.postMessage({
            type: "loadPose3D",
            pose3d: currentPose,
        }, "*");
    };

    const stopGraphZoom = (event) => {
        event.preventDefault();
        event.stopPropagation();
    };
    container.addEventListener("wheel", stopGraphZoom, { passive: false });
    iframe.addEventListener("wheel", stopGraphZoom, { passive: false });
    container.addEventListener("pointerdown", (event) => event.stopPropagation());
    container.addEventListener("pointermove", (event) => event.stopPropagation());

    const messageHandler = (event) => {
        if (event.source !== iframe.contentWindow) return;
        const data = event.data || {};
        if (data.type === "VIEWER_READY") {
            viewerReady = true;
            postPose();
        } else if (data.type === "POSE3D_CORRECTIONS") {
            const correctionWidget = getWidget(node, "corrections");
            if (correctionWidget) {
                const corrections = data.corrections && Object.keys(data.corrections).length
                    ? JSON.stringify(data.corrections)
                    : "";
                correctionWidget.value = corrections;
            }
            app.graph.setDirtyCanvas(true, false);
        } else if (data.type === "POSE3D_CAMERA") {
            // Editor camera -> the node's "camera" widget; render uses this viewpoint.
            const cameraWidget = getWidget(node, "camera");
            if (cameraWidget) {
                cameraWidget.value = data.camera ? JSON.stringify(data.camera) : "";
                app.graph.setDirtyCanvas(true, false);
            }
        }
    };
    window.addEventListener("message", messageHandler);

    const widget = node.addDOMWidget("preview", `pose3deditor${node.id}`, container, {
        getValue() {
            return "";
        },
        setValue() {},
    });

    widget.computeSize = function (width) {
        const w = Math.max(width || NODE_MIN_WIDTH, NODE_MIN_WIDTH);
        return [w, VIEWER_HEIGHT];
    };
    widget.element = container;

    if (typeof node.setSize === "function") {
        node.setSize([
            Math.max(node.size?.[0] ?? 0, NODE_MIN_WIDTH),
            Math.max(node.size?.[1] ?? 0, NODE_MIN_HEIGHT),
        ]);
    }

    const originalOnResize = node.onResize?.bind(node);
    node.onResize = function (size) {
        originalOnResize?.(size);
        if (size?.[1] && size[1] < NODE_MIN_HEIGHT && typeof node.setSize === "function") {
            node.setSize([Math.max(size[0] ?? NODE_MIN_WIDTH, NODE_MIN_WIDTH), NODE_MIN_HEIGHT]);
        }
        container.style.height = `${VIEWER_HEIGHT}px`;
    };

    widget.setPoseData = (pose3dData) => {
        currentPose = pose3dData?.joints_3d?.length ? {
            joints_3d: pose3dData.joints_3d.map((joint) => [...joint]),
            joint_names: pose3dData.joint_names?.length ? [...pose3dData.joint_names] : [...H36M_JOINT_NAMES],
            // SMPL-X editor mode: pass through so the viewer shows draggable joints
            // in TRUE metric space (normalizePose bypassed) with SMPL-X limbs + mesh.
            editorMode: !!pose3dData.editorMode,
            limbs: pose3dData.limbs || undefined,
            vertices: pose3dData.vertices || undefined,
            faces: pose3dData.faces || undefined,
            // skin weights drive the viewer's live soft-skinning (applySkinning);
            // without them the dragged mesh stays rigid.
            skin: pose3dData.skin || undefined,
            // which joint indices (0-54) get draggable handles (body + fingers).
            editable: pose3dData.editable || undefined,
        } : null;
        if (!currentPose) return;
        postPose();
    };

    const origOnRemoved = node.onRemoved?.bind(node);
    node.onRemoved = function () {
        origOnRemoved?.();
        viewerReady = false;
        window.removeEventListener("message", messageHandler);
    };

    requestAnimationFrame(() => node.onResize?.(node.size));
    return widget;
}

// ── ComfyUI extension ──────────────────────────────────────────────────────────

app.registerExtension({
    name: "editpose.ClickPoseEditor",

    nodeCreated(node) {
        // ── ClickPose — image preview + Edit Pose button ──────────────────────
        if (node.comfyClass !== "ClickPose") return;

        node._peKpsData = null;
        node._peImgUrl  = null;
        node._peImageId = null;

        // "Edit Pose" button
        const btn = node.addWidget("button", "Edit Pose", null, () => {
            if (!node._peKpsData || !node._peImgUrl) {
                alert("Queue the workflow first to detect the pose.");
                return;
            }
            openPoseEditorModal(node._peKpsData, node._peImgUrl, (corrections) => {
                const w = node.widgets?.find(w => w.name === "corrections");
                if (w) {
                    w.value = Object.keys(corrections).length > 0
                        ? JSON.stringify(corrections) : "";
                }
                // Patch local kps so next modal open starts from corrected positions
                for (const [idx, xy] of Object.entries(corrections)) {
                    const i = parseInt(idx);
                    node._peKpsData.keypoints[i] = xy;
                    node._peKpsData.scores[i] = 1.0;
                }
                app.graph.setDirtyCanvas(true, false);
            });
        });
        btn.serialize = false;

        // onExecuted — store kps state for editor modal, let ComfyUI handle the preview
        const origOnExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function (msg) {
            origOnExecuted?.(msg);
            if (!msg?.images?.length || !msg?.kps_json?.length) return;
            node._peImgUrl  = imageUrl(msg.images[0]);
            node._peKpsData = JSON.parse(msg.kps_json[0]);

            // Clear stale corrections when the image changes
            const incomingId = node._peKpsData.image_id;
            if (incomingId && incomingId !== node._peImageId) {
                node._peImageId = incomingId;
                const w = node.widgets?.find(w => w.name === "corrections");
                if (w) w.value = "";
            }
        };
    },
});

app.registerExtension({
    name: "editpose.Pose3DEditor",

    nodeCreated(node) {
        if (node.comfyClass !== "3D Pose Editor") return;

        node._pose3dData = null;
        const editorWidget = createEmbeddedNativePose3DEditor(node);

        const origOnExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function (msg) {
            origOnExecuted?.(msg);
            if (!msg?.pose3d_json?.length) return;
            node._pose3dData = JSON.parse(msg.pose3d_json[0]);
            const correctionWidget = getWidget(node, "corrections");
            if (correctionWidget) correctionWidget.value = "";
            editorWidget?.setPoseData(node._pose3dData);
        };
    },
});

app.registerExtension({
    name: "editpose.SMPLXEditor",

    nodeCreated(node) {
        if (node.comfyClass !== "SMPLXEditor") return;

        // Same embedded three.js viewer; setPoseData carries editorMode + limbs so
        // it renders draggable SMPL-X joints. Drags emit POSE3D_CORRECTIONS, which
        // the shared handler writes into the node's "corrections" widget; re-queue
        // re-solves body_pose to those targets.
        const editorWidget = createEmbeddedNativePose3DEditor(node);

        const origOnExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function (msg) {
            origOnExecuted?.(msg);
            if (!msg?.smplx_json?.length) return;
            const data = JSON.parse(msg.smplx_json[0]);
            // Clear corrections after a re-solve so the applied edit doesn't replay.
            const correctionWidget = getWidget(node, "corrections");
            if (correctionWidget) correctionWidget.value = "";
            editorWidget?.setPoseData(data);   // {joints_3d, joint_names, limbs, editorMode}
        };
    },
});
