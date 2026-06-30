import { app } from "../../scripts/app.js";
import { getViewerUrl } from "./utils/extensionFolder.js";

// ── helpers ─────────────────────────────────────────────────────────────────────

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) ?? null;
}

// ── embedded three.js SMPL-X editor (iframe viewer) ───────────────────────────────

function createEmbeddedNativePose3DEditor(node) {
    const NODE_MIN_WIDTH = 420;
    const CONTROLS_BAR = 76;          // MUST match #controls fixed height in viewer_pose3d.html
    const NODE_CHROME_HEIGHT = 130;   // node title + input widgets above the viewer
    // iframe height = width + bar  ->  the 3D view above the bar is exactly square (w x w).
    const iframeHeight = (w) => Math.max(w, NODE_MIN_WIDTH) + CONTROLS_BAR;
    const NODE_MIN_HEIGHT = iframeHeight(NODE_MIN_WIDTH) + NODE_CHROME_HEIGHT;

    const viewerUrl = getViewerUrl("viewer_pose3d");

    const container = document.createElement("div");
    Object.assign(container.style, {
        width: "100%",
        height: `${iframeHeight(NODE_MIN_WIDTH)}px`,
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
        const h = iframeHeight(w);            // height follows WIDTH -> square 3D view
        if (container) container.style.height = `${h}px`;
        return [w, h];
    };
    widget.element = container;

    if (typeof node.setSize === "function") {
        node.setSize([
            Math.max(node.size?.[0] ?? 0, NODE_MIN_WIDTH),
            NODE_MIN_HEIGHT,   // don't preserve a bloated saved height
        ]);
    }

    const originalOnResize = node.onResize?.bind(node);
    node.onResize = function (size) {
        originalOnResize?.(size);
        const w = Math.max(size?.[0] ?? NODE_MIN_WIDTH, NODE_MIN_WIDTH);
        container.style.height = `${iframeHeight(w)}px`;   // keep the 3D view square as width changes
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

    // An earlier resize-feedback bug could bloat the node height and persist it into
    // saved graphs. Force the node back to its natural compact size on load.
    requestAnimationFrame(() => {
        const w = Math.max(node.size?.[0] ?? NODE_MIN_WIDTH, NODE_MIN_WIDTH);
        container.style.height = `${iframeHeight(w)}px`;
        if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
            const nat = node.computeSize();
            node.setSize([Math.max(node.size?.[0] ?? nat[0], NODE_MIN_WIDTH), nat[1]]);
        }
        app.graph?.setDirtyCanvas?.(true, true);
    });
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
