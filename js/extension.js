import { app } from "../../scripts/app.js";
import { getViewerUrl } from "./utils/extensionFolder.js";

// ── helpers ─────────────────────────────────────────────────────────────────────

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) ?? null;
}

// ── embedded three.js SMPL-X editor (iframe viewer) ───────────────────────────────

function createEmbeddedNativePose3DEditor(node) {
    const DEFAULT_VIEWER_HEIGHT = 380;
    const MIN_VIEWER_HEIGHT = 260;
    const NODE_MIN_WIDTH = 320;
    const NODE_CHROME_HEIGHT = 130;
    const NODE_MIN_HEIGHT = MIN_VIEWER_HEIGHT + NODE_CHROME_HEIGHT;
    let viewerHeight = DEFAULT_VIEWER_HEIGHT;   // tracks node resize

    const viewerUrl = getViewerUrl("viewer_pose3d");

    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "100%",
        height: `${DEFAULT_VIEWER_HEIGHT}px`,
        minHeight: `${MIN_VIEWER_HEIGHT}px`,
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
        return [w, viewerHeight];
    };
    widget.element = container;

    if (typeof node.setSize === "function") {
        node.setSize([
            Math.max(node.size?.[0] ?? 0, NODE_MIN_WIDTH),
            Math.max(node.size?.[1] ?? 0, DEFAULT_VIEWER_HEIGHT + NODE_CHROME_HEIGHT),
        ]);
    }

    const originalOnResize = node.onResize?.bind(node);
    node.onResize = function (size) {
        originalOnResize?.(size);
        // viewer fills the node (minus the widgets/chrome above), down to a sane min,
        // so dragging the node resizes the 3D view instead of locking it at one height.
        const total = size?.[1] ?? NODE_MIN_HEIGHT;
        viewerHeight = Math.max(total - NODE_CHROME_HEIGHT, MIN_VIEWER_HEIGHT);
        container.style.height = `${viewerHeight}px`;
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
