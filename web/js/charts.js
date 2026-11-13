/**
 * Interactive Charts Engine using Chart.js
 * Renders Accuracy benchmarks, radar profile, and 20-epoch loss convergence
 */

let accuracyChart = null;
let radarChart = null;
let lossChart = null;
let edgeTradeoffChart = null;

function initCharts() {
    // 1. Grouped Accuracy Bar Chart
    const accEl = document.getElementById('accuracyChart');
    if (accEl) {
        const ctxAcc = accEl.getContext('2d');
        if (accuracyChart) accuracyChart.destroy();
        accuracyChart = new Chart(ctxAcc, {
            type: 'bar',
            data: {
                labels: ['Detection (mAP)', 'Pose (OKS AP)', 'Caption (CIDEr)', 'Syntax (BLEU-4)'],
                datasets: [
                    {
                        label: 'Track A (Monolithic Hydra)',
                        data: [3.7, 0.1, 46.2, 18.6],
                        backgroundColor: 'rgba(244, 63, 94, 0.85)',
                        borderColor: '#f43f5e',
                        borderWidth: 1.5,
                        borderRadius: 6
                    },
                    {
                        label: 'Track B (Modular Pipeline)',
                        data: [22.0, 45.0, 76.5, 24.2],
                        backgroundColor: 'rgba(16, 185, 129, 0.85)',
                        borderColor: '#10b981',
                        borderWidth: 1.5,
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1', font: { family: 'Inter', size: 12 } } },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 85,
                        grid: { color: 'rgba(51, 65, 85, 0.3)' },
                        ticks: { color: '#94a3b8', callback: (v) => v + '%' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    // 2. Capability Radar Chart
    const radarEl = document.getElementById('radarChart');
    if (radarEl) {
        const ctxRadar = radarEl.getContext('2d');
        if (radarChart) radarChart.destroy();
        radarChart = new Chart(ctxRadar, {
            type: 'radar',
            data: {
                labels: ['Spatial Localization', 'Skeletal Pose', 'Semantic Specificity', 'Syntactic Fluency', 'Inference Speed (FPS)', 'VRAM Efficiency'],
                datasets: [
                    {
                        label: 'Track A: Monolithic',
                        data: [17, 1, 60, 77, 95, 90],
                        backgroundColor: 'rgba(244, 63, 94, 0.25)',
                        borderColor: '#f43f5e',
                        pointBackgroundColor: '#f43f5e',
                        borderWidth: 2
                    },
                    {
                        label: 'Track B: Modular',
                        data: [85, 90, 95, 88, 40, 50],
                        backgroundColor: 'rgba(16, 185, 129, 0.25)',
                        borderColor: '#10b981',
                        pointBackgroundColor: '#10b981',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#cbd5e1' } } },
                scales: {
                    r: {
                        angleLines: { color: 'rgba(51, 65, 85, 0.5)' },
                        grid: { color: 'rgba(51, 65, 85, 0.5)' },
                        pointLabels: { color: '#cbd5e1', font: { size: 11, family: 'Inter' } },
                        ticks: { display: false }
                    }
                }
            }
        });
    }

    // 3. Loss Convergence Curves (Epoch 1 -> 20)
    const lossEl = document.getElementById('lossChart');
    if (lossEl) {
        const ctxLoss = lossEl.getContext('2d');
        if (lossChart) lossChart.destroy();
        
        const epochs = Array.from({ length: 20 }, (_, i) => `Ep ${i + 1}`);
        const boxLoss = [0.0191, 0.0135, 0.0112, 0.0097, 0.0084, 0.0072, 0.0066, 0.0059, 0.0053, 0.0048, 0.0046, 0.0042, 0.0039, 0.0036, 0.0035, 0.0033, 0.0032, 0.0030, 0.0029, 0.0028].map(v => v * 50);
        const poseLoss = [0.0756, 0.0648, 0.0600, 0.0561, 0.0529, 0.0500, 0.0475, 0.0449, 0.0420, 0.0396, 0.0375, 0.0355, 0.0334, 0.0314, 0.0300, 0.0281, 0.0269, 0.0256, 0.0244, 0.0231].map(v => v * 50);
        const capLoss = [3.2482, 2.5003, 2.2526, 2.0727, 1.9200, 1.7846, 1.6633, 1.5548, 1.4554, 1.3649, 1.2842, 1.2105, 1.1456, 1.0853, 1.0338, 0.9834, 0.9416, 0.9006, 0.8677, 0.8340];

        lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Weighted Box MSE (50x)',
                        data: boxLoss,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.08)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 3
                    },
                    {
                        label: 'Weighted Pose MSE (50x)',
                        data: poseLoss,
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.08)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 3
                    },
                    {
                        label: 'Caption Cross-Entropy (1x)',
                        data: capLoss,
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.08)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } },
                    tooltip: { mode: 'index', intersect: false }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: { color: 'rgba(51, 65, 85, 0.3)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(51, 65, 85, 0.2)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    // 4. Edge Latency vs VRAM Tradeoff
    const tradeEl = document.getElementById('tradeoffChart');
    if (tradeEl) {
        const ctxTrade = tradeEl.getContext('2d');
        if (edgeTradeoffChart) edgeTradeoffChart.destroy();
        edgeTradeoffChart = new Chart(ctxTrade, {
            type: 'bar',
            data: {
                labels: ['Inference Latency (ms)', 'Peak VRAM Footprint (GB)'],
                datasets: [
                    {
                        label: 'Track A (Monolithic)',
                        data: [12.4, 1.84],
                        backgroundColor: '#f43f5e',
                        borderRadius: 6
                    },
                    {
                        label: 'Track B (Modular Pipeline)',
                        data: [48.7, 4.12],
                        backgroundColor: '#10b981',
                        borderRadius: 6
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#cbd5e1' } }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(51, 65, 85, 0.3)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }
}

// Ensure charts re-render and scale properly when switching tabs
window.addEventListener('resize', () => {
    if (accuracyChart) accuracyChart.resize();
    if (radarChart) radarChart.resize();
    if (lossChart) lossChart.resize();
    if (edgeTradeoffChart) edgeTradeoffChart.resize();
});
