/**
 * Canvas Rendering Engine for Bounding Boxes and Skeletons
 * Provides smooth, high-DPI rendering for Track A & Track B predictions
 */

const COCO_BONES = [
    // Head
    [0, 1], [0, 2], [1, 3], [2, 4],
    // Upper body & arms
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    // Torso & Hips
    [5, 11], [6, 12], [11, 12],
    // Legs
    [11, 13], [13, 15], [12, 14], [14, 16]
];

const BONE_COLORS = {
    head: '#93c5fd',      // Light blue
    leftArm: '#38bdf8',   // Cyan
    rightArm: '#818cf8',  // Indigo
    torso: '#a855f7',     // Purple
    leftLeg: '#34d399',   // Emerald
    rightLeg: '#f59e0b'   // Amber
};

function getBoneColor(b1, b2, defaultColor = '#38bdf8') {
    if (b1 <= 4 && b2 <= 4) return BONE_COLORS.head;
    if ([5, 7, 9].includes(b1) && [5, 7, 9].includes(b2)) return BONE_COLORS.leftArm;
    if ([6, 8, 10].includes(b1) && [6, 8, 10].includes(b2)) return BONE_COLORS.rightArm;
    if ([11, 13, 15].includes(b1) && [11, 13, 15].includes(b2)) return BONE_COLORS.leftLeg;
    if ([12, 14, 16].includes(b1) && [12, 14, 16].includes(b2)) return BONE_COLORS.rightLeg;
    return defaultColor;
}

/**
 * Renders an image and its predicted annotations to an HTML5 canvas
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLImageElement} img
 * @param {Object} prediction - { box: [x, y, w, h], keypoints: [...], track: 'A'|'B', label: '...' }
 */
function renderPredictionToCanvas(canvas, img, prediction) {
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    
    // Set internal resolution matching original image for crisp rendering
    const origW = img.naturalWidth || img.width || 640;
    const origH = img.naturalHeight || img.height || 480;
    canvas.width = origW;
    canvas.height = origH;

    // Draw background image
    ctx.clearRect(0, 0, origW, origH);
    ctx.drawImage(img, 0, 0, origW, origH);

    if (!prediction) return;

    const isTrackA = prediction.track === 'A';
    const mainColor = isTrackA ? '#f43f5e' : '#10b981'; // Rose for A, Emerald for B
    const jointDotColor = isTrackA ? '#fb7185' : '#38bdf8';

    // 1. Draw Bounding Box
    if (prediction.box && prediction.box.length === 4) {
        let [bx, by, bw, bh] = prediction.box;
        
        // Guard against negative dimensions
        if (bw < 0) { bx = bx + bw; bw = Math.abs(bw); }
        if (bh < 0) { by = by + bh; bh = Math.abs(bh); }

        // Bounding Box Rectangle
        ctx.strokeStyle = mainColor;
        ctx.lineWidth = Math.max(3, Math.round(origW / 200));
        ctx.strokeRect(bx, by, bw, bh);

        // Label Tag Background
        const labelText = prediction.label || (isTrackA ? 'Track A: Monolithic Hydra' : 'Track B: SSDLite320 (Person)');
        ctx.font = `bold ${Math.max(12, Math.round(origW / 45))}px 'JetBrains Mono', monospace`;
        const textMetrics = ctx.measureText(labelText);
        const tagH = Math.max(22, Math.round(origW / 30));
        const tagW = textMetrics.width + 16;

        ctx.fillStyle = mainColor;
        ctx.fillRect(bx, Math.max(0, by - tagH), tagW, tagH);

        // Label Text
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(labelText, bx + 8, Math.max(tagH / 2, by - tagH / 2));
    }

    // 2. Draw Keypoints & Bones
    const kpts = prediction.keypoints || [];
    const parsedJoints = [];
    
    // Parse keypoints (format: [x1, y1, v1, x2, y2, v2, ...])
    for (let i = 0; i < kpts.length; i += 3) {
        parsedJoints.push({
            x: kpts[i],
            y: kpts[i + 1],
            v: kpts[i + 2]
        });
    }

    // Draw Skeleton Bones
    if (!isTrackA) {
        // Track B: draw anatomical skeleton connections
        COCO_BONES.forEach(([b1, b2]) => {
            const j1 = parsedJoints[b1];
            const j2 = parsedJoints[b2];
            if (j1 && j2 && (j1.v > 0) && (j2.v > 0)) {
                ctx.beginPath();
                ctx.moveTo(j1.x, j1.y);
                ctx.lineTo(j2.x, j2.y);
                ctx.strokeStyle = getBoneColor(b1, b2, mainColor);
                ctx.lineWidth = Math.max(3, Math.round(origW / 220));
                ctx.lineCap = 'round';
                ctx.stroke();
            }
        });
    }

    // Draw Joint Dots
    const dotRadius = Math.max(4, Math.round(origW / 140));
    parsedJoints.forEach((joint, idx) => {
        if (joint.v > 0 || isTrackA) {
            ctx.beginPath();
            ctx.arc(joint.x, joint.y, dotRadius, 0, 2 * Math.PI);
            ctx.fillStyle = jointDotColor;
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
    });
}
