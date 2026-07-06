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
    return "comfyui-smplx-estimator";
})();

/**
 * Build a cache-busted viewer URL, e.g. getViewerUrl("viewer_pose3d").
 *
 * Resolve it RELATIVE to this module's own URL (…/js/utils/extensionFolder.js) so it
 * works no matter what the install folder is named (comfyui-smplx-estimator from the
 * registry, ComfyUI-SMPLx-Estimator from a git-URL install, etc.) — no folder-name
 * guessing / hardcoded fallback.
 * @param {string} viewerName - viewer HTML filename without extension
 * @returns {string}
 */
export function getViewerUrl(viewerName) {
    try {
        const url = new URL(`../${viewerName}.html`, import.meta.url);
        return `${url.pathname}?v=${Date.now()}`;
    } catch (_) {
        return `/extensions/${EXTENSION_FOLDER}/${viewerName}.html?v=${Date.now()}`;
    }
}
