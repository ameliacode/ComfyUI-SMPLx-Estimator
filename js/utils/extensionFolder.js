/**
 * Auto-detect the extension folder name from import.meta.url, with a script-tag
 * fallback. Mirrors the comfy-3d-viewers convention so viewer URLs resolve the
 * same way regardless of how the package folder is named.
 */

export const EXTENSION_FOLDER = (() => {
    try {
        if (typeof import.meta !== "undefined" && import.meta.url) {
            const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
            if (m) return m[1];
        }
        const scripts = document.getElementsByTagName("script");
        for (let i = scripts.length - 1; i >= 0; i--) {
            const src = scripts[i].src;
            if (!src) continue;
            const m = src.match(/\/extensions\/([^/]+)\//);
            if (m) return m[1];
        }
    } catch (_) {
        // Best effort only.
    }
    return "comfyui-mocap";
})();

/**
 * Build a cache-busted viewer URL, e.g. getViewerUrl("viewer_pose3d").
 * @param {string} viewerName - viewer HTML filename without extension
 * @returns {string}
 */
export function getViewerUrl(viewerName) {
    return `/extensions/${EXTENSION_FOLDER}/${viewerName}.html?v=` + Date.now();
}
