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

scrambleBtn.addEventListener("click", scramble);
solveBtn.addEventListener("click", solve);
resetBtn.addEventListener("click", reset);

let currentCube = SOLVED;

async function scramble() {
    const response = await fetch("/api/scramble");
    const body = await response.json();

    renderCube(body.cube);
    scrambleMoves.textContent = body.moves.join(" ");
    solution.textContent = "";
    error.textContent = "";
}

async function solve() {
    const response = await fetch("/api/solve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cube: currentCube }),
    });
    const body = await response.json();

    if (!response.ok) {
        solution.textContent = "";
        error.textContent = errorMessage(body.detail);
        return;
    }

    error.textContent = "";
    if (body.moves.length === 0) {
        solution.textContent = "Already solved";
        return;
    }

    solution.textContent = body.moves.join(" ");
}

function errorMessage(detail) {
    if (!Array.isArray(detail)) {
        return String(detail);
    }
    return detail.map((item) => (typeof item === "string" ? item : item.msg)).join(" ");
}

function reset() {
    scrambleMoves.textContent = "";
    solution.textContent = "";
    error.textContent = "";

    renderCube(SOLVED);
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

renderCube(SOLVED);
