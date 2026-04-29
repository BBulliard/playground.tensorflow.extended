/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as nn from "./nn";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State, datasets, regDatasets, activations, problems,
  regularizations, getKeyFromValue, Problem
} from "./state";
import {ExampleND, shuffle, parseCSV,
        makeHyperplaneClassifier, makeHypersphereClassifier,
        toOneHot, discretizeLabels } from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

let mainWidth: number;

// ---------------------------------------------------------------------------
// Scroll helper
// ---------------------------------------------------------------------------
d3.select(".more button").on("click", function() {
  d3.transition().duration(1000).tween("scroll", scrollTween(800));
});
function scrollTween(offset: number) {
  return function() {
    const start = window.pageYOffset || document.documentElement.scrollTop;
    const i = d3.interpolateNumber(start, offset);
    return function(t: number) { scrollTo(0, i(t)); };
  };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

enum HoverType { BIAS, WEIGHT }

// ---------------------------------------------------------------------------
// InputFeature
// ---------------------------------------------------------------------------
interface InputFeature {
  f: (features: number[]) => number;
  label?: string;
}

const STATIC_2D_INPUTS: {[name: string]: InputFeature} = {
  "x":        {f: (v) => v[0],          label: "X_1"},
  "y":        {f: (v) => v[1],          label: "X_2"},
  "xSquared": {f: (v) => v[0] * v[0],   label: "X_1^2"},
  "ySquared": {f: (v) => v[1] * v[1],   label: "X_2^2"},
  "xTimesY":  {f: (v) => v[0] * v[1],   label: "X_1X_2"},
  "sinX":     {f: (v) => Math.sin(v[0]), label: "sin(X_1)"},
  "sinY":     {f: (v) => Math.sin(v[1]), label: "sin(X_2)"},
};
let INPUTS: {[name: string]: InputFeature} = {};

const HIDABLE_CONTROLS = [
  ["Show test data",      "showTestData"],
  ["Discretize output",   "discretize"],
  ["Play button",         "playButton"],
  ["Step button",         "stepButton"],
  ["Reset button",        "resetButton"],
  ["Learning rate",       "learningRate"],
  ["Activation",          "activation"],
  ["Regularization",      "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type",        "problem"],
  ["Which dataset",       "dataset"],
  ["Ratio train data",    "percTrainData"],
  ["Noise level",         "noise"],
  ["Batch size",          "batchSize"],
  ["# of hidden layers",  "numHiddenLayers"],
];

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------
class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: ((isPlaying: boolean) => void) | null = null;

  playOrPause() {
    if (this.isPlaying) { this.isPlaying = false; this.pause(); }
    else {
      this.isPlaying = true;
      if (iter === 0) simulationStarted();
      this.play();
    }
  }
  onPlayPause(callback: (isPlaying: boolean) => void) { this.callback = callback; }
  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) this.callback(this.isPlaying);
    this.start(this.timerIndex);
  }
  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) this.callback(this.isPlaying);
  }
  private start(localTimerIndex: number) {
    // d3 v7: d3.timer(fn) – the delay argument was removed; use setTimeout to replicate it
    setTimeout(() => {
      d3.timer(() => {
        if (localTimerIndex < this.timerIndex) return true;
        oneStep();
        return false;
      });
    }, 0);
  }
}

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
let state = State.deserializeState();
rebuildInputsFromState();

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string | null = null;
const xDomain: [number, number] = [-6, 6];

let heatMap = new HeatMap(
  300, DENSITY, xDomain, xDomain,
  d3.select("#heatmap") as unknown as d3.Selection<HTMLElement, unknown, null, undefined>,
  {showAxes: true}
);

// d3 v7: d3.scaleLinear() replaces d3.scale.linear()
const linkWidthScale = d3.scaleLinear().domain([0, 5]).range([1, 10]).clamp(true);
const colorScale = d3.scaleLinear<string>()
  .domain([-1, 0, 1]).range(["#f59322", "#e8eaeb", "#0877bd"]).clamp(true);

let iter = 0;
let trainData: ExampleND[] = [];
let testData: ExampleND[] = [];
let network: nn.Node[][] = [];
let lossTrain = 0;
let lossTest = 0;
const player = new Player();
const lineChart = new AppendingLineChart(
  d3.select("#linechart") as d3.Selection<HTMLElement, unknown, any, any>,
  ["#777", "black"]
);

// ---------------------------------------------------------------------------
// Input management
// ---------------------------------------------------------------------------
function rebuildInputsFromState(): void {
  INPUTS = {};
  if (state.isCSVDataset && state.activeInputs.length > 0) {
    state.selectedInputs.forEach((name) => {
      const idx = state.activeInputs.indexOf(name);
      INPUTS[name] = {f: (features) => features[idx], label: name};
    });
  } else {
    for (const name in STATIC_2D_INPUTS) {
      if (state[name]) INPUTS[name] = STATIC_2D_INPUTS[name];
    }
    if (Object.keys(INPUTS).length === 0) {
      INPUTS["x"] = STATIC_2D_INPUTS["x"];
      INPUTS["y"] = STATIC_2D_INPUTS["y"];
    }
  }
}

function setNDInputs(featureNames: string[]): void {
  state.activeInputs = featureNames.slice();
  state.currentFeatureNames = featureNames.slice();
  state.selectedInputs = featureNames.slice();
  INPUTS = {};
  featureNames.forEach((name, idx) => {
    INPUTS[name] = {f: (features) => features[idx], label: name};
  });
}

function constructInput(example: ExampleND): number[] {
  const input: number[] = [];
  for (const name in INPUTS) input.push(INPUTS[name].f(example.features));
  return input;
}
function constructInputIds(): string[] { return Object.keys(INPUTS); }

function getDisplayLabel(i: number): string {
  return (state.customClassLabels[i] && state.customClassLabels[i].trim() !== "")
    ? state.customClassLabels[i]
    : (state.classLabels[i] ?? String(i));
}

// ---------------------------------------------------------------------------
// GUI
// ---------------------------------------------------------------------------
function makeGUI() {
  // ------ switch modes -----------------------------------------------------
  d3.select("#mode-select").on("change", () => {
    const modeSelected = d3.select("#mode-select").property("value");
    state.isCSVDataset = modeSelected === "csv";

    if (modeSelected !== "csv") {
      state.numClasses = 1;     
      //state.csvData = null;       
      //state.activeInputs = []; 
      //state.selectedInputs = []; 
      rebuildInputsFromState();
    }else{
      if (state.csvRawText) {
        const isRegression = state.problem === Problem.REGRESSION;
        const { examples, featureNames, classLabels, numClasses } = parseCSV(
          state.csvRawText, state.csvLabelColumn ?? undefined, isRegression
        );
        state.csvData = examples;
        state.numClasses = numClasses;
        state.classLabels = classLabels;
        setNDInputs(featureNames);
      }
      generateData();
    } 
    updateOutputPanel();
    reset();
  });

  d3.select("#mode-select").property("value", state.isCSVDataset ? "csv" : "2d");

  //------- csv uploader -----------------------------------------------------
  const dropZone = document.getElementById("drop-zone");

  if (dropZone) {
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      const file = e.dataTransfer?.files[0];
      if (file){ 
        handleCSVFile(file);
        d3.select("#mode-select").property("value", "csv");
        (document.getElementById("mode-select") as HTMLSelectElement).dispatchEvent(new Event("change"));
      }
    });
  }

  const csvFileInput = document.getElementById("csv-file-input") as HTMLInputElement;
  csvFileInput.addEventListener("change", function() {
    const file = this.files && this.files[0];
    if (file){
      handleCSVFile(file);
      d3.select("#mode-select").property("value", "csv");
      (document.getElementById("mode-select") as HTMLSelectElement).dispatchEvent(new Event("change"));
    } 
    this.value = "";
  });

  document.getElementById("drop-zone")!
  .addEventListener("click", () => csvFileInput.click());

  //------- controls buttons -------------------------------------------------
  d3.select("#reset-button").on("click", () => { reset(); userHasInteracted(); });

  d3.select("#play-pause-button").on("click", () => { userHasInteracted(); player.playOrPause(); });
  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause(); userHasInteracted();
    if (iter === 0) simulationStarted();
    oneStep();
  });

  d3.select("#data-regen-button").on("click", () => { generateData(); parametersChanged = true; });

  // ---- ND generator dropdown -----------------------------------------------
  d3.select("#nd-generator").on("change", function(this: any) {
    const val = this.value;
    if (!val || val === "none") {
      state.isCSVDataset = false; 
      state.csvData = null; 
      state.activeInputs = [];
      rebuildInputsFromState();  
      generateData(); 
      parametersChanged = true; 
      reset();
      return;
    }
    const dim = parseInt(val, 10);
    if (isNaN(dim) || dim < 2) return;
    const featureNames: string[] = [];
    for (let i = 0; i < dim; i++) featureNames.push(`f${i}`);
    setNDInputs(featureNames);
    state.isCSVDataset = false; state.csvData = null;
    generateData(); 
    parametersChanged = true; 
    reset();
  });

  // ---- Dataset thumbnails --------------------------------------------------
  const dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function(this: any) {
    const newDataset = datasets[(this as any).dataset.dataset];
    if (newDataset === state.dataset) return;
    state.dataset = newDataset;
    state.isCSVDataset = false; state.csvData = null; state.activeInputs = [];
    rebuildInputsFromState();
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData(); parametersChanged = true; 
    reset();
  });
  d3.select(`canvas[data-dataset=${getKeyFromValue(datasets, state.dataset)}]`).classed("selected", true);

  const regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function(this: any) {
    const newDataset = regDatasets[(this as any).dataset.regdataset];
    if (newDataset === state.regDataset) return;
    state.regDataset = newDataset;
    state.isCSVDataset = false; state.csvData = null; state.activeInputs = [];
    rebuildInputsFromState();
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData(); parametersChanged = true; 
    reset();
  });
  d3.select(`canvas[data-regDataset=${getKeyFromValue(regDatasets, state.regDataset)}]`).classed("selected", true);

  // ---- Layer controls ------------------------------------------------------
  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) return;
    state.networkShape[state.numHiddenLayers] = 2; state.numHiddenLayers++;
    parametersChanged = true; reset();
  });
  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) return;
    state.numHiddenLayers--; state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true; reset();
  });

  const showTestData = d3.select("#show-test-data").on("change", function(this: any) {
    state.showTestData = this.checked; state.serialize(); userHasInteracted();
    if (!state.isCSVDataset) heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  showTestData.property("checked", state.showTestData);

  const discretize = d3.select("#discretize").on("change", function(this: any) {
    state.discretize = this.checked; state.serialize(); userHasInteracted(); updateUI();
  });
  discretize.property("checked", state.discretize);

  const percTrain = d3.select("#percTrainData").on("input", function(this: any) {
    state.percTrainData = +this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData(); parametersChanged = true; reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  const noise = d3.select("#noise").on("input", function(this: any) {
    state.noise = +this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    if (!state.isCSVDataset) { generateData(); parametersChanged = true; reset(); }
  });
  const currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) { noise.property("max", state.noise <= 80 ? state.noise : 50); if (state.noise > 80) state.noise = 50; }
  else if (state.noise < 0) state.noise = 0;
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  const batchSize = d3.select("#batchSize").on("input", function(this: any) {
    state.batchSize = +this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true; reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  d3.select("#activations").on("change", function(this: any) {
    state.activation = activations[this.value]; parametersChanged = true; reset();
  }).property("value", getKeyFromValue(activations, state.activation));

  d3.select("#learningRate").on("change", function(this: any) {
    state.learningRate = +this.value; state.serialize(); userHasInteracted(); parametersChanged = true;
  }).property("value", state.learningRate);

  d3.select("#regularizations").on("change", function(this: any) {
    state.regularization = regularizations[this.value]; parametersChanged = true; reset();
  }).property("value", getKeyFromValue(regularizations, state.regularization));

  d3.select("#regularRate").on("change", function(this: any) {
    state.regularizationRate = +this.value; parametersChanged = true; reset();
  }).property("value", state.regularizationRate);

  d3.select("#problem").on("change", function(this: any) {
    state.problem = (problems as any)[this.value]; generateData(); drawDatasetThumbnails(); parametersChanged = true; reset();
  }).property("value", getKeyFromValue(problems, state.problem));

  // d3 v7: d3.scaleLinear() + d3.axisBottom()
  const xColorScale = d3.scaleLinear().domain([-1, 1]).range([0, 144]);
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(d3.axisBottom(xColorScale).tickValues([-1, 0, 1]).tickFormat(d3.format("d") as any));

  window.addEventListener("resize", () => {
    const newWidth = document.querySelector("#main-part")!.getBoundingClientRect().width;
    if (newWidth !== mainWidth) { mainWidth = newWidth; drawNetwork(network); updateUI(true); }
  });

  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }

  // --- confusion matrix button -------------------------------------------
  d3.select("#confusion-matrix-button").on("click", () => {
  drawConfusionMatrix(testData);
});
}

// ---------------------------------------------------------------------------
// Heatmap / ND panel switching
// ---------------------------------------------------------------------------
function updateOutputPanel(): void {
  d3.select("#heatmap").style("display", state.isCSVDataset ? "none" : "block");
  d3.select("#heatmap-output-info").style("display", state.isCSVDataset ? "none" : "block");
  d3.selectAll(".ui-noise,.ui-dataset,.basic-button").style("display", state.isCSVDataset ? "none" : "block");
  d3.select("#csv-label-selector").style("display", state.isCSVDataset ? "block" : "none");

  const showAcc = state.isCSVDataset && state.problem === Problem.CLASSIFICATION;
  d3.select("#accuracy-panel").style("display", showAcc ? "block" : "none");
  d3.select("#confusion-matrix-button").style("display", showAcc ? "block" : "none");
}

// ---------------------------------------------------------------------------
// Network UI helpers
// ---------------------------------------------------------------------------
function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container: d3.Selection<SVGGElement, unknown, HTMLElement, any>) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    const currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      const node = currentLayer[i];

      if(state.isCSVDataset){
        const weightSum = node.inputLinks.reduce((sum, link) => sum + link.weight, 0);
        container.select(`#node${node.id} rect`)
          .style("fill", colorScale(weightSum));
      }

      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        // d3 v7: no more .style({}) object shorthand
        container.select(`#link${link.source.id}-${link.dest.id}`)
          .style("stroke-dashoffset", String(-iter / 3))
          .style("stroke-width", String(linkWidthScale(Math.abs(link.weight))))
          .style("stroke", colorScale(link.weight))
          .datum(link);
      }
    }
  }
}

function drawNode(
  cx: number, cy: number, nodeId: string, isInput: boolean,
  container: d3.Selection<SVGGElement, unknown, HTMLElement, any>,
  node?: nn.Node,
  isOutput = false
) {
  const x = cx - RECT_SIZE / 2;
  const y = cy - RECT_SIZE / 2;
  const isActive = state.isCSVDataset ? state.selectedInputs.includes(nodeId) : !!state[nodeId];
  const activeOrNotClass = isActive ? "active" : "inactive";

  if (isOutput && !state.isCSVDataset) return;

  // 1. svg group
  const nodeGroup = container.append("g")
    .attr("class", "node")
    .attr("id", `node${nodeId}`)
    .attr("transform", `translate(${x},${y})`);

  nodeGroup.append("rect")
    .attr("x", 0).attr("y", 0)
    .attr("width", RECT_SIZE)
    .attr("height", RECT_SIZE);

  if (isInput) { // 2. label des inputs
    const inputDef = INPUTS[nodeId] ?? STATIC_2D_INPUTS[nodeId];  // fix: fallback pour inputs inactifs
    const label = inputDef?.label ?? nodeId;
    const text = nodeGroup
      .append("text")
      .attr("class", "main-label").attr("x", -10)
      .attr("y", RECT_SIZE / 2).attr("text-anchor", "end");

    if (!state.isCSVDataset) { // 2a. regex notation mathématique (2D uniquement)
      if (/[_^]/.test(label)) {
        const myRe = /(.*?)([_^])(.)/g;
        let myArray: RegExpExecArray | null; let lastIndex = 0;
        while ((myArray = myRe.exec(label)) != null) {
          lastIndex = myRe.lastIndex;
          if (myArray[1]) text.append("tspan").text(myArray[1]);
          text.append("tspan")
            .attr("baseline-shift", myArray[2] === "_" ? "sub" : "super")
            .style("font-size", "9px").text(myArray[3]);
        }
        if (label.substring(lastIndex)) text.append("tspan").text(label.substring(lastIndex));
      } else {
        text.append("tspan").text(label);
      }
    } else { // 2b. sliced label and correlation calculation
      const parts = label.split("_");
      const maxChars = 15;
      const lineHeight = 11;
      const totalHeight = (parts.length - 1) * lineHeight;
      parts.forEach((part, i) => {
        if (part.length > maxChars) {
          part.slice(0, maxChars - 1) + "…";
        }
        text.append("tspan")
          .attr("x", -10)
          .attr("dy", i === 0 ? -totalHeight / 2 : lineHeight)
          .text(part);
      });

      const idx = state.activeInputs.indexOf(nodeId);
      if (idx !== -1 && trainData.length > 0) {
        const n = trainData.length;
        const meanX = trainData.reduce((s, d) => s + d.features[idx], 0) / n;
        const meanY = trainData.reduce((s, d) => s + d.label, 0) / n;
        const num   = trainData.reduce((s, d) => s + (d.features[idx] - meanX) * (d.label - meanY), 0);
        const denX  = Math.sqrt(trainData.reduce((s, d) => s + Math.pow(d.features[idx] - meanX, 2), 0));
        const denY  = Math.sqrt(trainData.reduce((s, d) => s + Math.pow(d.label - meanY, 2), 0));
        const corr  = (denX * denY) > 0 ? num / (denX * denY) : 0;
        nodeGroup.select("rect").style("fill", colorScale(corr));
      }
    }
    nodeGroup.classed(activeOrNotClass, true);

  } else if (!isOutput){ // 3. biais (ND et 2D)
    nodeGroup.append("rect")
      .attr("id", `bias-${nodeId}`).attr("x", -BIAS_SIZE - 2)
      .attr("y", RECT_SIZE - BIAS_SIZE + 3).attr("width", BIAS_SIZE).attr("height", BIAS_SIZE)
      .on("mouseenter", function(event: MouseEvent) {
        updateHoverCard(HoverType.BIAS, node, d3.pointer(event, container.node()));
      })
      .on("mouseleave", () => updateHoverCard(null));
  }

  if (state.isCSVDataset) { // 4. mode CSV
    if (!isInput) {
      // 4a. canvas pour nœuds cachés ND uniquement
      const canvasDiv = (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
        .insert("div", ":first-child")
        .attr("id", `canvas-${nodeId}`).attr("class", "canvas")
        .style("position", "absolute")
        .style("left", `${x + 3}px`)
        .style("top", `${y + 3}px`);
      if (node) {
        const weightSum = node.inputLinks.reduce((sum, link) => sum + link.weight, 0);
        nodeGroup.select("rect").style("fill", colorScale(weightSum));
      }
      if (isOutput) {
        const outputIndex = network[network.length - 1].findIndex(n => n.id === nodeId);
        const labelText = nodeGroup
          .append("text")
          .attr("class", "main-label output-class-label")
          .attr("x", RECT_SIZE + 10).attr("y", RECT_SIZE / 2)
          .attr("text-anchor", "start")
          .style("cursor", "pointer")
          .style("dominant-baseline", "middle");
        labelText.append("tspan").text(getDisplayLabel(outputIndex));
        labelText.append("title").text("Click to rename this class");

        labelText.on("click", function(event: MouseEvent) {
          event.stopPropagation();

          const textNode = labelText.node() as SVGTextElement;
          const rect = textNode.getBoundingClientRect();

          const inp = document.createElement("input");
          inp.type = "text";
          inp.value = (state.customClassLabels[outputIndex] !== "")
            ? state.customClassLabels[outputIndex]
            : (state.classLabels[outputIndex] ?? String(outputIndex));

          inp.className = "output-label-editor";
          inp.style.left   = rect.left + "px";
          inp.style.top    = rect.top  + "px";
          inp.style.height = Math.max(rect.height, 18) + "px";

          document.body.appendChild(inp);
          inp.focus();
          inp.select();

          function commit() {
            const newVal = inp.value.trim();
            state.customClassLabels[outputIndex] = newVal;
            labelText.select("tspan")
              .text(newVal !== "" ? newVal : (state.classLabels[outputIndex] ?? String(outputIndex)));
            inp.remove();
          }

          inp.addEventListener("blur", commit);
          inp.addEventListener("keydown", (e: KeyboardEvent) => {
            if (e.key === "Enter")  { commit(); }
            if (e.key === "Escape") { inp.remove(); }
          });
        });    
      }
    } else {
      // 4b. clic sur le groupe SVG pour activer/désactiver un input CSV
      nodeGroup.style("cursor", "pointer")
        .on("click", () => {
          const idx = state.selectedInputs.indexOf(nodeId);
          if (idx === -1) {
            state.selectedInputs.push(nodeId);
          } else {
            if (state.selectedInputs.length === 1) return;
            state.selectedInputs.splice(idx, 1);
          }
          rebuildInputsFromState();
          parametersChanged = true;
          reset();
        });
    }
    return;
  } else { // 5. mode 2D clickable canvas and mini-heatmap
    if (isOutput) return;
    // 5a. canvas pour tous les nœuds 2D non-output
    const canvasDiv = (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
      .insert("div", ":first-child")
      .attr("id", `canvas-${nodeId}`).attr("class", "canvas")
      .style("position", "absolute").style("left", `${x + 3}px`).style("top", `${y + 3}px`)
      .on("mouseenter", function() {
        selectedNodeId = nodeId;
        canvasDiv.classed("hovered", true);
        nodeGroup.classed("hovered", true);
        updateDecisionBoundary(network, false);
        if (boundary[nodeId]){
          heatMap.updateBackground(boundary[nodeId], state.discretize);
        }
      })
      .on("mouseleave", function() {
        selectedNodeId = null;
        canvasDiv.classed("hovered", false);
        nodeGroup.classed("hovered", false);
        updateDecisionBoundary(network, false);
        if (boundary[nodeId]) {
          heatMap.updateBackground(boundary[nn.getOutputNode(network).id], state.discretize);
        }
      });

    if (isInput) {
      // 5b. clic pour activer/désactiver un input 2D
      canvasDiv.on("click", () => {
        state[nodeId] = !state[nodeId];
        rebuildInputsFromState();
        parametersChanged = true;
        reset();
      });
      canvasDiv.style("cursor", "pointer").classed(activeOrNotClass, true);
    }

    if (isInput) {
      canvasDiv.classed(activeOrNotClass, true);
    } 

    // 5c. mini-heatmap pour tous les nœuds 2D non-output
    if (!state.isCSVDataset && !isOutput) {
      const nodeHeatMap = new HeatMap(
        RECT_SIZE, DENSITY / 10, xDomain, xDomain,
        canvasDiv as unknown as d3.Selection<HTMLElement, unknown, null, undefined>,
        {noSvg: true}
      );
      canvasDiv.datum({heatmap: nodeHeatMap, id: nodeId});
    }
  }
}

// ---------------------------------------------------------------------------
// Network drawing
// ---------------------------------------------------------------------------
function drawNetwork(network: nn.Node[][]): void {
  const svg = d3.select("#svg") as d3.Selection<SVGSVGElement, unknown, HTMLElement, any>;
  svg.select("g.core").remove();
  (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
    .selectAll("div.canvas").remove();
  (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
    .selectAll("div.plus-minus-neurons").remove();

  const padding = 3;
  const co = d3.select(".column.output").node() as HTMLDivElement;
  const cf = d3.select(".column.features").node() as HTMLDivElement;
  const width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  const node2coord: {[id: string]: {cx: number, cy: number}} = {};
  const container = svg.append("g").classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);

  const numLayers = network.length;
  const featureWidth = 118;

  // d3 v7: d3.scalePoint() replaces d3.scale.ordinal().rangePoints()
  const layerScale = d3.scalePoint<number>()
    .domain(d3.range(1, numLayers - 1))
    .range([featureWidth, width - RECT_SIZE])
    .padding(0.7);

  const nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);

  const calloutThumb  = d3.select(".callout.thumbnail").style("display", "none");
  const calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout: string | null = null;
  let targetIdWithCallout: string | null = null;

  // Input layer
  const baseCx = RECT_SIZE / 2 + 50;
  const nodeIds = state.isCSVDataset ? state.activeInputs : Object.keys(STATIC_2D_INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    const cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx: baseCx, cy};
    drawNode(baseCx, cy, nodeId, true, container as any);
  });

  // Hidden layers
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    const numNodes = network[layerIdx].length;
    const cx = layerScale(layerIdx)! + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx)!, layerIdx);
    for (let i = 0; i < numNodes; i++) {
      const node = network[layerIdx][i];
      const cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container as any, node);
      const nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null && i === numNodes - 1 && nextNumNodes <= numNodes) {
        calloutThumb.style("display", null).style("top", `${20 + 3 + cy}px`).style("left", `${cx}px`);
        idWithCallout = node.id;
      }
      for (let j = 0; j < node.inputLinks.length; j++) {
        const link = node.inputLinks[j];
        const path = drawLink(link, node2coord, network, container as any, j === 0, j, node.inputLinks.length)
          .node() as SVGPathElement;
        const prevLayer = network[layerIdx - 1];
        const lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null && i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout && prevLayer.length >= numNodes) {
          const midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style("display", null).style("top", `${midPoint.y + 5}px`).style("left", `${midPoint.x + 3}px`);
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Output nodes
  const outputCx = width + RECT_SIZE / 2;
  const outputLayer = network[numLayers - 1];
  outputLayer.forEach((outputNode, i) => {
    const outputCy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[outputNode.id] = {cx: outputCx, cy: outputCy};
    drawNode(outputCx, outputCy, outputNode.id, false, container as any, outputNode, true);
    for (let j = 0; j < outputNode.inputLinks.length; j++) {
      drawLink(outputNode.inputLinks[j], node2coord, network, container as any, j === 0, j, outputNode.inputLinks.length);
    }
  });

  svg.attr("height", maxY);
  const height = Math.max(
    getRelativeHeight(calloutThumb as any),
    getRelativeHeight(calloutWeights as any),
    getRelativeHeight(d3.select("#network") as any)
  );
  d3.select(".column.features").style("height", height + "px");
  d3.select("#accuracy-panel").style("margin-top",  outputLayer.length*(RECT_SIZE+30) + "px");
}

function getRelativeHeight(selection: d3.Selection<HTMLElement, unknown, any, any>): number {
  const node = selection.node() as HTMLElement;
  return node.offsetHeight + node.offsetTop;
}

function addPlusMinusControl(x: number, layerIdx: number) {
  const div = (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
    .append("div").classed("plus-minus-neurons", true).style("left", `${x - 10}px`);
  const i = layerIdx - 1;
  const firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button").attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => { if (state.networkShape[i] >= 8) return; state.networkShape[i]++; parametersChanged = true; reset(); })
    .append("i").attr("class", "material-icons").text("add");
  firstRow.append("button").attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => { if (state.networkShape[i] <= 1) return; state.networkShape[i]--; parametersChanged = true; reset(); })
    .append("i").attr("class", "material-icons").text("remove");
  div.append("div").text(state.networkShape[i] + " neuron" + (state.networkShape[i] > 1 ? "s" : ""));
}

function updateHoverCard(type: HoverType | null, nodeOrLink?: nn.Node | nn.Link, coordinates?: [number, number]) {
  const hovercard = d3.select("#hovercard");
  if (type == null) { hovercard.style("display", "none"); d3.select("#svg").on("click", null); return; }

  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    const input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function(this: any) {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) (nodeOrLink as nn.Link).weight = +this.value;
        else (nodeOrLink as nn.Node).bias = +this.value;
        updateUI();
      }
    });
    // d3 v7: event is passed directly as first argument to the handler
    input.on("keypress", function(event: KeyboardEvent) {
      if (event.keyCode === 13) updateHoverCard(type, nodeOrLink, coordinates);
    });
    (input.node() as HTMLInputElement).focus();
  });

  const value = type === HoverType.WEIGHT ? (nodeOrLink as nn.Link).weight : (nodeOrLink as nn.Node).bias;
  // d3 v7: no more .style({}) object shorthand
  hovercard.style("left", `${coordinates![0] + 20}px`).style("top", `${coordinates![1]}px`).style("display", "block");
  hovercard.select(".type").text(type === HoverType.WEIGHT ? "Weight" : "Bias");
  hovercard.select(".value").style("display", null).text(value.toPrecision(2));
  hovercard.select("input").property("value", value.toPrecision(2)).style("display", "none");
}

// ---------------------------------------------------------------------------
// drawLink – d3.svg.diagonal() was removed in d3 v5; replaced by cubic Bezier
// ---------------------------------------------------------------------------
function drawLink(
  link: nn.Link,
  node2coord: {[id: string]: {cx: number, cy: number}},
  network: nn.Node[][],
  container: d3.Selection<SVGGElement, unknown, HTMLElement, any>,
  isFirst: boolean, index: number, length: number
) {
  const source = node2coord[link.source.id];
  const dest   = node2coord[link.dest.id];
  const x1 = source.cx + RECT_SIZE / 2 + 2;
  const y1 = source.cy;
  const x2 = dest.cx - RECT_SIZE / 2;
  const y2 = dest.cy + ((index - (length - 1) / 2) / length) * 12;
  const dx = (x2 - x1) / 2;
  const pathD = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;

  const line = container.insert("path", ":first-child")
    .attr("marker-start", "url(#markerArrow)")
    .attr("class", "link")
    .attr("id", "link" + link.source.id + "-" + link.dest.id)
    .attr("d", pathD);

  // d3 v7: event handler receives (event, datum) as arguments
  container.append("path").attr("d", pathD).attr("class", "link-hover")
    .on("mouseenter", function(event: MouseEvent) {
      updateHoverCard(HoverType.WEIGHT, link, d3.pointer(event, container.node()));
    })
    .on("mouseleave", () => updateHoverCard(null));

  return line;
}

// ---------------------------------------------------------------------------
// Decision boundary (2-D mode only)
// ---------------------------------------------------------------------------
/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in STATIC_2D_INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scaleLinear([0, DENSITY - 1], xDomain);
  let yScale = d3.scaleLinear([DENSITY - 1, 0], xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in STATIC_2D_INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      nn.forwardProp(network, constructInput({features: [x, y], label: 0}));
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in STATIC_2D_INPUTS) {
          boundary[nodeId][i][j] = STATIC_2D_INPUTS[nodeId].f([x, y]);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Loss / Accuracy / UI / training / confusion matrix
// ---------------------------------------------------------------------------
function getLoss(network: nn.Node[][], dataPoints: ExampleND[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    const outputs = nn.forwardProp(network, constructInput(dataPoints[i]));
    if (state.isCSVDataset) {
      const targets = toOneHot(dataPoints[i].label, state.numClasses);
      targets.forEach((t, k) => {
        loss += nn.Errors.CROSS_ENTROPY.error(outputs[k], t);
      });
    } else {
      loss += nn.Errors.SQUARE.error(outputs[0], dataPoints[i].label);
    }
  }
  return loss / dataPoints.length;
}

function getAccuracy(network: nn.Node[][], dataPoints: ExampleND[]): number {
  if (dataPoints.length === 0) return NaN;
  if (state.problem !== Problem.CLASSIFICATION) return NaN;
 
  let correct = 0;
  for (const point of dataPoints) {
    const outputs = nn.forwardProp(network, constructInput(point));
    const predicted = outputs.indexOf(Math.max(...outputs));
    if (predicted === point.label) correct++;
  }
  return correct / dataPoints.length;
}

function updateUI(firstStep = false) {
  updateWeightsUI(network, d3.select("g.core") as any);
  updateBiasesUI(network);

  if (!state.isCSVDataset) {
    updateDecisionBoundary(network, firstStep);
    const selectedId = selectedNodeId != null ? selectedNodeId : nn.getOutputNode(network).id;
    heatMap.updateBackground(boundary[selectedId], state.discretize);
    (d3.select("#network") as d3.Selection<HTMLElement, unknown, HTMLElement, any>)
      .selectAll<HTMLDivElement, {heatmap: HeatMap, id: string}>("div.canvas")
      .each(function(data) {
      if (boundary[data.id]) {
        data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10), state.discretize);
      }
    });
  } else if (state.isCSVDataset && state.problem === Problem.CLASSIFICATION) {
    const accTrain = getAccuracy(network, trainData);
    const accTest  = getAccuracy(network, testData);
    const fmt = (v: number) => isNaN(v) ? "—" : (v * 100).toFixed(1) + "%";
    const pct = (v: number) => isNaN(v) ? 0 : Math.round(v * 100);
    d3.select("#acc-train").text(fmt(accTrain));
    d3.select("#acc-test").text(fmt(accTest));
    d3.select("#acc-bar-train").style("width", pct(accTrain) + "%");
    d3.select("#acc-bar-test").style("width", pct(accTest) + "%");
  }  
  
  function zeroPad(n: number): string { const pad = "000000"; return (pad + n).slice(-pad.length); }
  function addCommas(s: string): string { return s.replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  d3.select("#loss-train").text(lossTrain.toFixed(3));
  d3.select("#loss-test").text(lossTest.toFixed(3));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    nn.forwardProp(network, constructInput(point));
    
    if (state.isCSVDataset && state.numClasses > 1) {
      // Mode multi-classes/CSV
      const targets = toOneHot(point.label, state.numClasses);
      nn.backProp(network, targets, nn.Errors.CROSS_ENTROPY);
    } else {
      // Mode 2D
      nn.backProp(network, [point.label], nn.Errors.SQUARE);
    }

    if ((i + 1) % state.batchSize === 0)
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
  });
  lossTrain = getLoss(network, trainData);
  lossTest  = getLoss(network, testData);
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  const weights: number[] = [];
  for (let l = 0; l < network.length - 1; l++)
    for (let i = 0; i < network[l].length; i++)
      for (let j = 0; j < network[l][i].outputs.length; j++)
        weights.push(network[l][i].outputs[j].weight);
  return weights;
}

function computeConfusionMatrix(data: ExampleND[]): number[][] {
  const n = state.numClasses;
  const matrix = Array.from({length: n}, () => new Array(n).fill(0));
  for (const point of data) {
    const outputs = nn.forwardProp(network, constructInput(point));
    const predicted = outputs.indexOf(Math.max(...outputs));
    matrix[point.label][predicted]++;
  }
  return matrix;
}

function drawConfusionMatrix(data: ExampleND[]): void {
  const matrix = computeConfusionMatrix(data);
  const labels = state.classLabels; 
  const n = state.numClasses;
  const cellSize = 48;
  const margin = { top: 60, left: 70, right: 10, bottom: 10 };
  const svgW = margin.left + n * cellSize + margin.right;
  const svgH = margin.top  + n * cellSize + margin.bottom;
  const maxCount = Math.max(...matrix.flat());

  // Color scale: white (0) → blue (max) for correct, white → orange for errors
  const opacityScale = d3.scaleLinear().domain([0, maxCount]).range([0, 1]);

  // Clear any previous matrix and (re)create the SVG ---
  const container = d3.select("#confusion-matrix-container");
  container.html("")

  const svg = container.append("svg")
    .attr("width", svgW)
    .attr("height", svgH);

  const g = svg.append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);

  matrix.forEach((row, i) => {           // i = actual class
    row.forEach((count, j) => {          // j = predicted class
      const isCorrect = (i === j);
      const baseColor = isCorrect ? "#0877bd" : "#f59322";

      g.append("rect")
        .attr("x", j * cellSize)
        .attr("y", i * cellSize)
        .attr("width", cellSize - 2)     
        .attr("height", cellSize - 2)
        .attr("rx", 3)                   
        .style("fill", baseColor)
        .style("opacity", 0.15 + opacityScale(count) * 0.85);

      g.append("text")
        .attr("x", j * cellSize + cellSize / 2 - 1)
        .attr("y", i * cellSize + cellSize / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "13px")
        .style("font-weight", isCorrect ? "500" : "300")
        .style("fill", opacityScale(count) > 0.5 ? "white" : "#333")
        .text(count);
    });
  });

  // Column labels
  labels.forEach((label, j) => {
    g.append("text")
      .attr("x", j * cellSize + cellSize / 2 - 1)
      .attr("y", -8)
      .attr("text-anchor", "middle")
      .style("font-size", "11px")
      .style("fill", "#777")
      .text(label);
  });

  // "Predicted" axis title
  g.append("text")
    .attr("x", (n * cellSize) / 2)
    .attr("y", -40)
    .attr("text-anchor", "middle")
    .style("font-size", "12px")
    .style("fill", "#333")
    .style("font-weight", "500")
    .text("Predicted");

  // Row labels
  labels.forEach((label, i) => {
    g.append("text")
      .attr("x", -8)
      .attr("y", i * cellSize + cellSize / 2)
      .attr("text-anchor", "end")
      .attr("dominant-baseline", "middle")
      .style("font-size", "11px")
      .style("fill", "#777")
      .text(label);
  });

  // "Actual" axis title
  g.append("text")
    .attr("transform", `translate(-50, ${(n * cellSize) / 2}) rotate(-90)`)
    .attr("text-anchor", "middle")
    .style("font-size", "12px")
    .style("fill", "#333")
    .style("font-weight", "500")
    .text("Actual");
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------
function reset(onStartup = false) {
  lineChart.reset();
  state.serialize();

  if (!onStartup) userHasInteracted();
  player.pause();
  selectedNodeId = null;
  d3.select("#layers-label").text("Hidden layer" + (state.numHiddenLayers !== 1 ? "s" : ""));
  d3.select("#num-layers").text(state.numHiddenLayers);
  iter = 0;
  
  const numInputs = Object.keys(INPUTS).length;
  const shape = [numInputs].concat(state.networkShape).concat([state.numClasses]);
  const outputActivation = (state.problem === Problem.REGRESSION) 
    ? nn.Activations.LINEAR  
    : (state.numClasses > 1 ? nn.Activations.SOFTMAX : nn.Activations.TANH);

  network = nn.buildNetwork(shape, state.activation, outputActivation,
    state.regularization, constructInputIds(), state.initZero);

  lossTrain = getLoss(network, trainData);
  lossTest  = getLoss(network, testData);

  drawNetwork(network);
  updateOutputPanel();
  updateUI(true);
}

function initTutorial() {
  if (state.tutorial == null || state.tutorial === "" || state.hideText) return;
  d3.selectAll("article div.l--body").remove();
  const tutorial = d3.select("article").append("div").attr("class", "l--body");
  fetch(`tutorials/${state.tutorial}.html`)
    .then(r => r.text())
    .then(html => {
      const node = tutorial.node() as HTMLElement;
      node.innerHTML = html;
      const titleEl = node.querySelector("title");
      if (titleEl) {
        d3.select("header h1").style("margin-top", "20px").style("margin-bottom", "20px").text(titleEl.textContent);
        document.title = titleEl.textContent;
      }
    })
    .catch(err => console.error("Tutorial load error:", err));
}

// ---------------------------------------------------------------------------
// Dataset thumbnails
// ---------------------------------------------------------------------------
function drawDatasetThumbnails() {
  function renderThumbnail(canvas: HTMLCanvasElement, gen: (n: number, noise: number) => ExampleND[]) {
    canvas.setAttribute("width", "100"); canvas.setAttribute("height", "100");
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    gen(200, 0).forEach((d: ExampleND) => {
      const px = d.features[0] || 0, py = d.features[1] || 0;
      ctx.fillStyle = colorScale(d.label);
      ctx.fillRect(100 * (px + 6) / 12, 100 * (py + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode as HTMLElement).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");
  if (state.problem === Problem.CLASSIFICATION) {
    for (const ds in datasets) {
      const c = document.querySelector(`canvas[data-dataset=${ds}]`) as HTMLCanvasElement;
      if (c) renderThumbnail(c, datasets[ds]);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (const ds in regDatasets) {
      const c = document.querySelector(`canvas[data-regDataset=${ds}]`) as HTMLCanvasElement;
      if (c) renderThumbnail(c, regDatasets[ds]);
    }
  }
}

// ---------------------------------------------------------------------------
// Hide controls
// ---------------------------------------------------------------------------
function hideControls() {
  const hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    const controls = d3.selectAll(`.ui-${prop}`);
    if ((controls as any).size() === 0) console.warn(`0 html elements found with class .ui-${prop}`);
    controls.style("display", "none");
  });
  const hcContainer = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    const label = hcContainer.append("label").attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    const input = label.append("input").attr("type", "checkbox").attr("class", "mdl-checkbox__input");
    if (hiddenProps.indexOf(id) === -1) input.attr("checked", "true");
    input.on("change", function(this: any) {
      state.setHideProperty(id, !this.checked); state.serialize(); userHasInteracted();
      d3.select(".hide-controls-link").attr("href", window.location.href);
    });
    label.append("span").attr("class", "mdl-checkbox__label label").text(text);
  });
  d3.select(".hide-controls-link").attr("href", window.location.href);
}

// ---------------------------------------------------------------------------
// Data generation
// ---------------------------------------------------------------------------
function generateData(firstTime = false) {
  if (!firstTime) {
    state.seed = Math.random().toFixed(5); 
    state.serialize(); 
    userHasInteracted(); 
  }

  Math.seedrandom(state.seed);
  let allData: ExampleND[];
  if (state.isCSVDataset && state.csvData != null) {
    allData = state.csvData.slice();
  } else if (state.isCSVDataset && state.activeInputs.length > 2) {
    const dim = state.activeInputs.length;
    const numSamples = state.problem === Problem.REGRESSION ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    const generator = state.problem === Problem.REGRESSION
      ? makeHyperplaneClassifier(dim) : makeHypersphereClassifier(dim);
    allData = generator(numSamples, state.noise / 100);
    if (allData.length > 0 && allData[0].featureNames) setNDInputs(allData[0].featureNames);
  } else {
    const numSamples = state.problem === Problem.REGRESSION ? NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
    allData = (state.problem === Problem.CLASSIFICATION ? state.dataset : state.regDataset)(numSamples, state.noise / 100);
  }
  shuffle(allData);
  const splitIndex = Math.floor(allData.length * state.percTrainData / 100);
  trainData = allData.slice(0, splitIndex);
  testData  = allData.slice(splitIndex);
  if (!state.isCSVDataset) {
    heatMap.updatePoints(trainData);
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  }
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------
let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) return;
  firstInteraction = false;
  let page = "index";
  if (state.tutorial != null && state.tutorial !== "") page = `/v/tutorials/${state.tutorial}`;
  ga("set", "page", page);
  ga("send", "pageview", {"sessionControl": "start"});
}
function simulationStarted() {
  ga("send", {hitType: "event", eventCategory: "Starting Simulation",
    eventAction: parametersChanged ? "changed" : "unchanged",
    eventLabel: state.tutorial == null ? "" : state.tutorial});
  parametersChanged = false;
}

// ---------------------------------------------------------------------------
// Label selector
// ---------------------------------------------------------------------------
function buildLabelSelector(): void {
  const labelSelect = d3.select("#csv-label-select");
  labelSelect.html("");

  state.csvAllColumns.forEach(col => {
    labelSelect.append("option")
      .attr("value", col)
      .property("selected", col === state.csvLabelColumn)
      .text(col);
  });

  labelSelect.on("change", function() {
    const maxClasses = 8;
    state.csvLabelColumn = (this as HTMLSelectElement).value;
    const isRegression = (state.problem === Problem.REGRESSION); 
    const parsed = parseCSV(state.csvRawText!, state.csvLabelColumn, isRegression);
    let examples = parsed.examples;
    let numClasses = parsed.numClasses;
    const featureNames = parsed.featureNames;
    const classLabels = parsed.classLabels;
    let bins = 0;

    let discretizedClassLabels = classLabels;
    if (!isRegression && numClasses > maxClasses) {
      while (bins < 2 || bins > maxClasses){
        bins = parseInt(
          prompt(
            `La colonne "${state.csvLabelColumn}" contient ${numClasses} valeurs uniques (max ${maxClasses}).\n` +
            `En combien de classes voulez-vous la discrétiser ?`,
            "5"
          ) ?? "5"
        );
        if (isNaN(bins) || bins < 2 || bins > maxClasses) {
          alert(`Nombre de classes invalide. Entrez un nombre entre 2 et ${maxClasses}.`);
          continue;
        }
        ({ examples, binLabels: discretizedClassLabels } = discretizeLabels(parsed.rawExamples, bins));
        numClasses = bins;   
      }
    }

    state.csvData = examples;
    state.numClasses = numClasses;
    state.classLabels = discretizedClassLabels;
    state.customClassLabels = new Array(numClasses).fill("");
    setNDInputs(featureNames);
    buildLabelSelector();   
    generateData();
    parametersChanged = true;
    reset();
  });
}
// ---------------------------------------------------------------------------
// File Handler
// ---------------------------------------------------------------------------
function handleCSVFile(file: File): void {
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const text = (e.target as FileReader).result as string;
      const isRegression = (state.problem === Problem.REGRESSION);

      // 1. read all columns
      const allColumns = text.split(/\r?\n/)[0].split(",").map(c => c.trim());
      state.csvAllColumns = allColumns;
      state.csvRawText = text;

      // 2. define default label
      if (!state.csvLabelColumn || !allColumns.includes(state.csvLabelColumn)) {
        state.csvLabelColumn = allColumns[allColumns.length - 1];
      }

      // 3. parse label
      const { examples, featureNames, classLabels, numClasses } = parseCSV(text, state.csvLabelColumn, isRegression);

      // 4. update state
      state.csvData = examples;
      state.isCSVDataset = true;
      state.numClasses = numClasses;
      state.classLabels = classLabels;

      // 5. construct ui
      setNDInputs(featureNames);
      buildLabelSelector();
      d3.selectAll(".ui-modeSelect").style("display","block");
      d3.select("#mode-select").property("value", "csv")
      d3.select("#csv-label-selector").style("display", "block");
      generateData();
      parametersChanged = true;
      reset();

      d3.select("#drop-zone-text")
      .html(`📄 ${file.name} <br> (${examples.length} rows, <br> ${featureNames.length} features)`);

      if(!state.csvLabelColumn || !allColumns.includes(state.csvLabelColumn)) {
        state.csvLabelColumn = allColumns[allColumns.length-1];
      }
    } catch (err) {
      alert(`CSV parsing error: ${(err as Error).message}`);
    }
  };
  reader.readAsText(file);
}
// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();