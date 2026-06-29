import { app } from "../../scripts/app.js";

// ── helpers ─────────────────────────────────────────────────────────────────────

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

// ── embedded three.js SMPL-X editor (iframe viewer) ───────────────────────────────

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
        iframe.contentWindow.postMessage({ type: "loadPose3D", pose3d: currentPose }, "*");
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
                correctionWidget.value = data.corrections && Object.keys(data.corrections).length
                    ? JSON.stringify(data.corrections)
                    : "";
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
        getValue() { return ""; },
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
            joint_names: [...(pose3dData.joint_names || [])],
            // editor mode: draggable joints in TRUE metric space + SMPL-X mesh.
            editorMode: !!pose3dData.editorMode,
            limbs: pose3dData.limbs || undefined,
            vertices: pose3dData.vertices || undefined,
            faces: pose3dData.faces || undefined,
            // skin weights drive the viewer's live soft-skinning (applySkinning).
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

// ── ComfyUI extension: SMPL-X Editor ─────────────────────────────────────────────

app.registerExtension({
    name: "editpose.SMPLXEditor",

    nodeCreated(node) {
        if (node.comfyClass !== "SMPLXEditor") return;

        // Embedded three.js viewer. setPoseData carries editorMode + limbs + skin +
        // editable so it renders draggable SMPL-X joints (body + fingers). Drags emit
        // POSE3D_CORRECTIONS -> the node's "corrections" widget; re-queue re-solves.
        const editorWidget = createEmbeddedNativePose3DEditor(node);

        const origOnExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function (msg) {
            origOnExecuted?.(msg);
            if (!msg?.smplx_json?.length) return;
            const data = JSON.parse(msg.smplx_json[0]);
            const correctionWidget = getWidget(node, "corrections");
            if (correctionWidget) correctionWidget.value = "";   // clear after a re-solve
            editorWidget?.setPoseData(data);
        };
    },
});
