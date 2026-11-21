const COLORS = {"U": "#ffffff", "R": "#d62d2d", "F": "#2a9d4a", "D": "#f5d021", "L": "#f2841c", "B": "#2463c9"};
const SOLVED = "U".repeat(9) + "R".repeat(9) + "F".repeat(9) + "D".repeat(9) + "L".repeat(9) + "B".repeat(9);
const DEFAULT_ROT_X = -25;
const DEFAULT_ROT_Y = -35;
const FACE_ORDER = ["U", "R", "F", "D", "L", "B"];
const COORDS = [-1, 0, 1];
const DEFAULT_TURN_MS = 300;
const REDUCED_MOTION_TURN_MS = 150;
const HALF_TURN_FACTOR = 1.5;

const FACE_POSITIONS = {
    U: (row, col) => [col, -1, row],
    R: (row, col) => [1, row, -col],
    F: (row, col) => [col, row, 1],
    D: (row, col) => [col, 1, -row],
    L: (row, col) => [-1, row, col],
    B: (row, col) => [-col, row, -1],
};

const TURNS = {
    U: { axis: "Y", coord: "y", side: -1, angle: -90 },
    D: { axis: "Y", coord: "y", side: 1, angle: 90 },
    R: { axis: "X", coord: "x", side: 1, angle: 90 },
    L: { axis: "X", coord: "x", side: -1, angle: -90 },
    F: { axis: "Z", coord: "z", side: 1, angle: 90 },
    B: { axis: "Z", coord: "z", side: -1, angle: -90 },
};

const scrambleBtn = document.getElementById("scramble-btn");
const solveBtn = document.getElementById("solve-btn");
const resetBtn = document.getElementById("reset-btn");
const scrambleMoves = document.getElementById("scramble-moves");
const solution = document.getElementById("solution");
const error = document.getElementById("error");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const stepLabel = document.getElementById("step");
const methodSelect = document.getElementById("method-select");
const solveInfo = document.getElementById("solve-info");
const scene = document.getElementById("cube-scene");
const cube3d = document.getElementById("cube-3d");
const palette = document.getElementById("palette");
const turnTimeSlider = document.getElementById("turn-time");
const turnTimeValue = document.getElementById("turn-time-value");

scrambleBtn.addEventListener("click", scramble);
solveBtn.addEventListener("click", solve);
resetBtn.addEventListener("click", reset);
prevBtn.addEventListener("click", previous);
nextBtn.addEventListener("click", next);
turnTimeSlider.addEventListener("input", showTurnTime);
scene.addEventListener("pointerdown", onPointerDown);
scene.addEventListener("pointermove", onPointerMove);
scene.addEventListener("pointerup", onPointerUp);
scene.addEventListener("pointercancel", onPointerUp);

let currentCube = SOLVED;
let moves = [];
let step = 0;
let busy = false;
let rotX = DEFAULT_ROT_X;
let rotY = DEFAULT_ROT_Y;
let dragging = false;
let lastX = 0;
let lastY = 0;
let brush = "U";
let downTarget = null;
let startX = 0;
let startY = 0;

function buildPalette() {
    FACE_ORDER.forEach((letter) => {
        const button = document.createElement("button");
        button.type = "button";
        button.classList.add("color-btn");
        button.dataset.color = letter;
        button.style.backgroundColor = COLORS[letter];
        button.setAttribute("aria-label", `Paint with the ${letter} face colour`);
        button.addEventListener("click", () => selectColor(letter));
        palette.append(button);
    });
}

function selectColor(letter) {
    brush = letter;
    for (const button of palette.children) {
        button.classList.toggle("selected", button.dataset.color === letter);
    }
}

function paintSticker(index) {
    if (busy || solveBtn.disabled || index % 9 === 4) {
        return;
    }

    const cubeString = currentCube.substring(0, index) + brush + currentCube.substring(index + 1);
    renderCube(cubeString);
    scrambleMoves.textContent = "";
    clearSolution();
}

function clearSolution() {
    solveInfo.textContent = "";
    solution.textContent = "";
    error.textContent = "";
    moves = [];
    step = 0;
    stepLabel.textContent = "";
}

async function scramble() {
    if (busy) {
        return;
    }

    const response = await fetch("/api/scramble");
    const body = await response.json();

    renderCube(body.cube);
    scrambleMoves.textContent = body.moves.join(" ");
    clearSolution();
}

async function loadMethods() {
    const response = await fetch("/api/methods");
    const body = await response.json();

    body.forEach((method) => {
        const option = document.createElement("option");
        option.value = method.name;
        option.textContent = method.label;
        methodSelect.append(option);
    });
}

async function solve() {
    if (busy) {
        return;
    }

    solveBtn.disabled = true;
    solveBtn.textContent = "Solving…";
    try {
        const response = await fetch("/api/solve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ cube: currentCube, method: methodSelect.value }),
        });
        const body = await response.json();

        if (!response.ok) {
            solution.textContent = "";
            solveInfo.textContent = "";
            error.textContent = errorMessage(body.detail);
            return;
        }

        error.textContent = "";
        if (body.moves.length === 0) {
            solution.textContent = "Already solved";
            solveInfo.textContent = "";
            return;
        }

        moves = body.moves;
        step = 0;
        solveInfo.textContent = `${body.length} moves in ${body.time_ms} ms`;

        renderSolution();
    } finally {
        solveBtn.disabled = false;
        solveBtn.textContent = "Solve";
    }
}

function errorMessage(detail) {
    if (!Array.isArray(detail)) {
        return String(detail);
    }
    return detail.map((item) => (typeof item === "string" ? item : item.msg)).join(" ");
}

function reset() {
    if (busy) {
        return;
    }

    scrambleMoves.textContent = "";
    clearSolution();
    renderCube(SOLVED);
}

async function applyMove(move) {
    const response = await fetch("/api/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cube: currentCube, moves: [move] }),
    });
    const body = await response.json();

    if (!response.ok) {
        error.textContent = errorMessage(body.detail);
        return false;
    }

    error.textContent = "";
    try {
        await animateTurn(move);
    } finally {
        renderCube(body.cube);
    }
    return true;
}

async function animateTurn(move) {
    const turn = TURNS[move[0]];
    let angle = turn.angle;
    if (move.endsWith("'")) {
        angle = -angle;
    }
    if (move.endsWith("2")) {
        angle *= 2;
    }

    const layer = document.createElement("div");
    layer.classList.add("layer");
    for (const cubie of [...cube3d.children]) {
        if (Number(cubie.dataset[turn.coord]) === turn.side) {
            layer.append(cubie);
        }
    }
    cube3d.append(layer);

    let duration = Number(turnTimeSlider.value);
    if (move.endsWith("2")) {
        duration *= HALF_TURN_FACTOR;
    }
    const keyframes = [
        { transform: `rotate${turn.axis}(0deg)` },
        { transform: `rotate${turn.axis}(${angle}deg)` },
    ];
    await layer.animate(keyframes, { duration, easing: "ease-in-out", fill: "forwards" }).finished;
}

async function previous() {
    if (busy || step === 0) {
        return;
    }

    busy = true;
    try {
        if (await applyMove(inverse(moves[step - 1]))) {
            step--;
            renderSolution();
        }
    } finally {
        busy = false;
    }
}

async function next() {
    if (busy || step === moves.length) {
        return;
    }

    busy = true;
    try {
        if (await applyMove(moves[step])) {
            step++;
            renderSolution();
        }
    } finally {
        busy = false;
    }
}

function inverse(move) {
    if (move.endsWith("'")) {
        return move.slice(0, -1);
    }

    if (move.endsWith("2")) {
        return move;
    }

    return move + "'";
}

function renderCube(cubeString) {
    currentCube = cubeString;

    const cubies = new Map();
    for (const x of COORDS) {
        for (const y of COORDS) {
            for (const z of COORDS) {
                cubies.set(`${x},${y},${z}`, buildCubie(x, y, z));
            }
        }
    }

    for (let index = 0; index < cubeString.length; index++) {
        const letter = FACE_ORDER[Math.floor(index / 9)];
        const [x, y, z] = stickerPosition(index);
        const plane = cubies.get(`${x},${y},${z}`).querySelector(`.face-${letter}`);
        plane.append(buildSticker(index, cubeString[index]));
    }

    cube3d.replaceChildren(...cubies.values());
}

function stickerPosition(index) {
    const letter = FACE_ORDER[Math.floor(index / 9)];
    const row = Math.floor((index % 9) / 3) - 1;
    const col = (index % 3) - 1;
    return FACE_POSITIONS[letter](row, col);
}

function buildCubie(x, y, z) {
    const cubie = document.createElement("div");
    cubie.classList.add("cubie");
    cubie.dataset.x = x;
    cubie.dataset.y = y;
    cubie.dataset.z = z;
    cubie.style.setProperty("--x", x);
    cubie.style.setProperty("--y", y);
    cubie.style.setProperty("--z", z);

    FACE_ORDER.forEach((letter) => {
        const plane = document.createElement("div");
        plane.classList.add("cubie-face", `face-${letter}`);
        cubie.append(plane);
    });

    return cubie;
}

function buildSticker(index, letter) {
    const sticker = document.createElement("div");
    sticker.classList.add("sticker");
    sticker.style.backgroundColor = COLORS[letter];
    sticker.dataset.index = index;
    if (index % 9 === 4) {
        sticker.classList.add("center");
    }
    return sticker;
}

function applyRotation() {
    cube3d.style.transform = `rotateX(${rotX}deg) rotateY(${rotY}deg)`;
}

function onPointerDown(event) {
    dragging = true;
    downTarget = event.target;
    startX = lastX = event.clientX;
    startY = lastY = event.clientY;
    scene.setPointerCapture(event.pointerId);
}

function onPointerMove(event) {
    if (!dragging) {
        return;
    }

    rotY += (event.clientX - lastX) * 0.5;
    rotX = clamp(rotX - (event.clientY - lastY) * 0.5, -90, 90);
    lastX = event.clientX;
    lastY = event.clientY;

    applyRotation();
}

function onPointerUp(event) {
    dragging = false;

    const distance = Math.hypot(event.clientX - startX, event.clientY - startY);
    if (event.type === "pointerup" && distance < 5 && downTarget.dataset.index !== undefined) {
        paintSticker(Number(downTarget.dataset.index));
    }
    downTarget = null;
}

function renderSolution() {
    solution.textContent = "";

    for (let i = 0; i < moves.length; i++) {
        const chip = document.createElement("span");
        chip.classList.add("chip");

        if (i < step) {
            chip.classList.add("done");
        }

        if (i === step) {
            chip.classList.add("current");
        }

        chip.textContent = moves[i];
        solution.append(chip);
    }

    stepLabel.textContent = `Step ${step} of ${moves.length}`;
}

function setDefaultTurnTime() {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    turnTimeSlider.value = reducedMotion ? REDUCED_MOTION_TURN_MS : DEFAULT_TURN_MS;
    showTurnTime();
}

function showTurnTime() {
    turnTimeValue.textContent = `${turnTimeSlider.value} ms`;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
}

renderCube(SOLVED);
applyRotation();
buildPalette();
selectColor(brush);
setDefaultTurnTime();
loadMethods();
