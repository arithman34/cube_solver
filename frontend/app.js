const COLORS = {"U": "#ffffff", "R": "#d62d2d", "F": "#2a9d4a", "D": "#f5d021", "L": "#f2841c", "B": "#2463c9"};
const FACE_OFFSETS = [[0, 3], [3, 6], [3, 3], [6, 3], [3, 0], [3, 9]];
const SOLVED = "U".repeat(9) + "R".repeat(9) + "F".repeat(9) + "D".repeat(9) + "L".repeat(9) + "B".repeat(9);

const net = document.getElementById("cube-net");
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

scrambleBtn.addEventListener("click", scramble);
solveBtn.addEventListener("click", solve);
resetBtn.addEventListener("click", reset);
prevBtn.addEventListener("click", previous);
nextBtn.addEventListener("click", next);

let currentCube = SOLVED;
let moves = [];
let step = 0;
let busy = false;

async function scramble() {
    const response = await fetch("/api/scramble");
    const body = await response.json();

    renderCube(body.cube);
    scrambleMoves.textContent = body.moves.join(" ");
    solveInfo.textContent = "";
    solution.textContent = "";
    error.textContent = "";
    moves = [];
    step = 0;
    stepLabel.textContent = "";
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
    scrambleMoves.textContent = "";
    solveInfo.textContent = "";
    solution.textContent = "";
    error.textContent = "";
    moves = [];
    step = 0;
    stepLabel.textContent = "";

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
    renderCube(body.cube);
    return true;
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
    net.innerHTML = "";
    currentCube = cubeString;
    for (let i = 0; i < 54; i++) {
        const face = Math.floor(i / 9);
        const row = Math.floor((i % 9) / 3);
        const col = i % 3;
        
        const sticker = document.createElement("div");
        sticker.classList.add("sticker");
        sticker.style.backgroundColor = COLORS[cubeString[i]];
        sticker.style.gridRow = FACE_OFFSETS[face][0] + row + 1;
        sticker.style.gridColumn = FACE_OFFSETS[face][1] + col + 1;

        net.append(sticker);
    }
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

renderCube(SOLVED);
loadMethods();
