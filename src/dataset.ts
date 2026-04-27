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

import * as d3 from 'd3';

/**
 * A N-dimensional example: an array of feature values and a label.
 * Replaces the former Example2D type.
 *
 * For built-in 2D generators the features array always has the form
 * [x, y], so existing rendering code that accesses features[0] / features[1]
 * keeps working without change.
 */
export type ExampleND = {
  features: number[];   // length == number of input dimensions
  label: number;
  /** Human-readable column names, parallel to `features`. */
  featureNames?: string[];
};

/**
 * Legacy alias kept so that any external code still importing Example2D
 * continues to compile.  New code should use ExampleND directly.
 */
export type Example2D = ExampleND;

/** Internal 2-D point helper – not exported. */
type Point2D = { x: number; y: number };

// ---------------------------------------------------------------------------
// Shuffle
// ---------------------------------------------------------------------------

/**
 * Shuffles the array in-place using the Fisher-Yates algorithm.
 * Uses Math.random (which may be seeded via seedrandom).
 */
export function shuffle(array: any[]): void {
  let counter = array.length;
  while (counter > 0) {
    const index = Math.floor(Math.random() * counter);
    counter--;
    const temp = array[counter];
    array[counter] = array[index];
    array[index] = temp;
  }
}

// ---------------------------------------------------------------------------
// DataGenerator type
// ---------------------------------------------------------------------------

/**
 * A function that produces a dataset.
 * Synchronous generators (built-in) simply ignore the third argument.
 * The optional `featureNames` field on the returned examples is populated
 * by generators that know their column names (e.g. CSV loader).
 */
export type DataGenerator = (numSamples: number, noise: number) => ExampleND[];

// ---------------------------------------------------------------------------
// Helper: wrap a (x,y) pair into ExampleND
// ---------------------------------------------------------------------------

function point2D(x: number, y: number, label: number): ExampleND {
  return { features: [x, y], label, featureNames: ['x', 'y'] };
}

// ---------------------------------------------------------------------------
// Built-in 2-D generators (unchanged logic, new return type)
// ---------------------------------------------------------------------------

export function classifyTwoGaussData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];

  const varianceScale = d3.scaleLinear().domain([0, 0.5]).range([0.5, 4]);
  const variance = varianceScale(noise);

  function genGauss(cx: number, cy: number, label: number) {
    for (let i = 0; i < numSamples / 2; i++) {
      const x = normalRandom(cx, variance);
      const y = normalRandom(cy, variance);
      points.push(point2D(x, y, label));
    }
  }

  genGauss(2, 2, 1);
  genGauss(-2, -2, -1);
  return points;
}

export function regressPlane(numSamples: number, noise: number): ExampleND[] {
  const radius = 6;
  const labelScale = d3.scaleLinear().domain([-10, 10]).range([-1, 1]);
  const getLabel = (x: number, y: number) => labelScale(x + y);

  const points: ExampleND[] = [];
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-radius, radius);
    const y = randUniform(-radius, radius);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getLabel(x + noiseX, y + noiseY);
    points.push(point2D(x, y, label));
  }
  return points;
}

export function regressGaussian(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];

  const labelScale = d3.scaleLinear().domain([0, 2]).range([1, 0]).clamp(true);

  const gaussians: [number, number, number][] = [
    [-4, 2.5, 1], [0, 2.5, -1], [4, 2.5, 1],
    [-4, -2.5, -1], [0, -2.5, 1], [4, -2.5, -1],
  ];

  function getLabel(x: number, y: number) {
    let label = 0;
    gaussians.forEach(([cx, cy, sign]) => {
      const newLabel = sign * labelScale(dist({ x, y }, { x: cx, y: cy }));
      if (Math.abs(newLabel) > Math.abs(label)) {
        label = newLabel;
      }
    });
    return label;
  }

  const radius = 6;
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-radius, radius);
    const y = randUniform(-radius, radius);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getLabel(x + noiseX, y + noiseY);
    points.push(point2D(x, y, label));
  }
  return points;
}

export function classifySpiralData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const n = numSamples / 2;

  function genSpiral(deltaT: number, label: number) {
    for (let i = 0; i < n; i++) {
      const r = (i / n) * 5;
      const t = 1.75 * (i / n) * 2 * Math.PI + deltaT;
      const x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      const y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push(point2D(x, y, label));
    }
  }

  genSpiral(0, 1);
  genSpiral(Math.PI, -1);
  return points;
}

export function classifySimpleData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const length = 5;
  const getLabel = (p: Point2D) => (p.y > 0 ? 1 : -1);

  for (let i = 0; i < numSamples / 64; i++) {
    const x = randUniform(-length, length) + randUniform(-1, 1) * noise;
    const y = randUniform(0.25, length) + randUniform(-1, 1) * noise;
    points.push(point2D(x, y, getLabel({ x, y })));
  }
  for (let i = 0; i < numSamples / 64; i++) {
    const x = randUniform(-length, length) + randUniform(-1, 1) * noise;
    const y = randUniform(-length, -0.25) + randUniform(-1, 1) * noise;
    points.push(point2D(x, y, getLabel({ x, y })));
  }
  return points;
}

export function classifyCircleData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const radius = 5;
  const getLabel = (p: Point2D, center: Point2D) =>
    dist(p, center) < radius * 0.5 ? 1 : -1;

  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(0, radius * 0.5);
    const angle = randUniform(0, 2 * Math.PI);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push(point2D(x, y, label));
  }
  for (let i = 0; i < numSamples / 2; i++) {
    const r = randUniform(radius * 0.7, radius);
    const angle = randUniform(0, 2 * Math.PI);
    const x = r * Math.sin(angle);
    const y = r * Math.cos(angle);
    const noiseX = randUniform(-radius, radius) * noise;
    const noiseY = randUniform(-radius, radius) * noise;
    const label = getLabel({ x: x + noiseX, y: y + noiseY }, { x: 0, y: 0 });
    points.push(point2D(x, y, label));
  }
  return points;
}

export function classifyXORData(numSamples: number, noise: number): ExampleND[] {
  const getLabel = (p: Point2D) => (p.x * p.y >= 0 ? 1 : -1);
  const points: ExampleND[] = [];
  for (let i = 0; i < numSamples; i++) {
    const padding = 0.3;
    let x = randUniform(-5, 5);
    x += x > 0 ? padding : -padding;
    let y = randUniform(-5, 5);
    y += y > 0 ? padding : -padding;
    const noiseX = randUniform(-5, 5) * noise;
    const noiseY = randUniform(-5, 5) * noise;
    const label = getLabel({ x: x + noiseX, y: y + noiseY });
    points.push(point2D(x, y, label));
  }
  return points;
}

export function classifyDiagonalStripesData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const size = 5;
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-size, size);
    const y = randUniform(-size, size);
    const nx = randUniform(-1, 1) * noise * size;
    const ny = randUniform(-1, 1) * noise * size;
    const val = Math.sin(((x + nx) - (y + ny)) * Math.PI / 3);
    const label = val >= 0 ? 1 : -1;
    points.push(point2D(x, y, label));
  }
  return points;
}

export function classifyTwoMoonsData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const n = numSamples / 2;
  for (let i = 0; i < n; i++) {
    const angle = Math.PI * (i / n);
    const r = 3;
    const x =  r * Math.cos(angle) + randUniform(-1, 1) * noise;
    const y =  r * Math.sin(angle) + randUniform(-1, 1) * noise;
    points.push(point2D(x - 1.5, y - 1, 1));
  }
  for (let i = 0; i < n; i++) {
    const angle = Math.PI * (i / n);
    const r = 3;
    const x =  r * Math.cos(angle + Math.PI) + randUniform(-1, 1) * noise;
    const y =  r * Math.sin(angle + Math.PI) + randUniform(-1, 1) * noise;
    points.push(point2D(x + 1.5, y + 1, -1));
  }
  return points;
}

export function classifyCheckerboardData(numSamples: number, noise: number): ExampleND[] {
  const points: ExampleND[] = [];
  const size = 5;
  for (let i = 0; i < numSamples; i++) {
    const x = randUniform(-size, size);
    const y = randUniform(-size, size);
    const nx = randUniform(-size, size) * noise;
    const ny = randUniform(-size, size) * noise;
    const col = Math.floor((x + nx + size) / 2.5);
    const row = Math.floor((y + ny + size) / 2.5);
    const label = (col + row) % 2 === 0 ? 1 : -1;
    points.push(point2D(x, y, label));
  }
  return points;
}

// ---------------------------------------------------------------------------
// N-dimensional built-in generators
// ---------------------------------------------------------------------------

/**
 * Classifies points in N-D space using a hyperplane decision boundary.
 * Label = sign(w · features) where w is a random unit vector fixed at
 * construction time (but re-drawn each call for reproducibility with seed).
 *
 * featureNames: ['f0', 'f1', ..., 'f{dim-1}']
 */
export function makeHyperplaneClassifier(dim: number): DataGenerator {
  return function classifyHyperplaneND(numSamples: number, noise: number): ExampleND[] {
    // Random weight vector (will be consistent when Math.random is seeded)
    const w: number[] = [];
    for (let d = 0; d < dim; d++) {
      w.push(Math.random() * 2 - 1);
    }
    const wNorm = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
    const wUnit = w.map(v => v / wNorm);

    const featureNames = Array.from({ length: dim }, (_, i) => `f${i}`);
    const points: ExampleND[] = [];

    for (let i = 0; i < numSamples; i++) {
      const features = Array.from({ length: dim }, () => randUniform(-5, 5));
      const noisy = features.map(f => f + randUniform(-5, 5) * noise);
      const dot = noisy.reduce((s, f, d) => s + f * wUnit[d], 0);
      const label = dot >= 0 ? 1 : -1;
      points.push({ features, label, featureNames });
    }
    return points;
  };
}

/**
 * Classifies points in N-D space using a hypersphere decision boundary.
 * Label = 1 if ||features|| < radius/2, else -1.
 *
 * featureNames: ['f0', 'f1', ..., 'f{dim-1}']
 */
export function makeHypersphereClassifier(dim: number): DataGenerator {
  return function classifyHypersphereND(numSamples: number, noise: number): ExampleND[] {
    const radius = 5;
    const featureNames = Array.from({ length: dim }, (_, i) => `f${i}`);
    const points: ExampleND[] = [];

    for (let i = 0; i < numSamples; i++) {
      const features = Array.from({ length: dim }, () => randUniform(-radius, radius));
      const noisy = features.map(f => f + randUniform(-radius, radius) * noise);
      const normSq = noisy.reduce((s, f) => s + f * f, 0);
      const label = Math.sqrt(normSq) < radius * 0.5 ? 1 : -1;
      points.push({ features, label, featureNames });
    }
    return points;
  };
}

// ---------------------------------------------------------------------------
// CSV loader
// ---------------------------------------------------------------------------

/**
 * Parse a raw CSV string into an ExampleND array.
 *
 * Rules:
 *  - First row must be a header with column names.
 *  - The label column is identified by `labelCol` (default: last column).
 *  - All other numeric columns become features.
 *  - Non-numeric cells are skipped (the entire row is dropped).
 *  - Label values: any value ≤ 0 becomes -1, any value > 0 becomes +1
 *    (classification). For regression pass `isRegression=true` to keep
 *    the raw numeric value.
 *
 * @returns { examples, featureNames } so callers can update INPUTS dynamically.
 */
export function parseCSV(
  csvText: string,
  labelCol?: string,
  isRegression = false,
): { examples: ExampleND[]; rawExamples: ExampleND[]; featureNames: string[]; classLabels: string[], numClasses: number } {
  const lines = csvText.trim().split(/\r?\n/);
  if (lines.length < 2) {
    throw new Error('CSV must have at least a header row and one data row.');
  }

  // Parse header
  const header = splitCSVLine(lines[0]);

  // Determine label column index
  let labelIdx: number;
  if (labelCol != null) {
    labelIdx = header.indexOf(labelCol);
    if (labelIdx === -1) {
      throw new Error(`Label column "${labelCol}" not found in CSV header.`);
    }
  } else {
    labelIdx = header.length - 1; // default: last column
  }

  const featureIndices: number[] = [];
  const featureNames: string[] = [];
  header.forEach((name, i) => {
    if (i !== labelIdx) {
      featureIndices.push(i);
      featureNames.push(name.trim());
    }
  });

  const examples: ExampleND[] = [];

  for (let r = 1; r < lines.length; r++) {
    const line = lines[r].trim();
    const cells = splitCSVLine(line);
    const rawValues = cells.map(c => parseFloat(c.trim()));
    const features = featureIndices.map(i => rawValues[i]);
    const rawLabel = rawValues[labelIdx];
    const label = rawLabel;

    examples.push({ features, label, featureNames });
  }

  if (examples.length === 0) {
    throw new Error('CSV contained no valid numeric rows after parsing.');
  }

  const classLabels = [...new Set(examples.map(e => String(e.label)))].sort();
  const labelToIndex = new Map(classLabels.map((l, i) => [l, i]));
  const remapped = examples.map(e => ({
    ...e,
    label: labelToIndex.get(String(e.label))!
  }));
return { examples: remapped, rawExamples: examples, featureNames, classLabels, numClasses: classLabels.length };
}

/** Minimal CSV line splitter (handles quoted fields with commas). */
function splitCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

// ---------------------------------------------------------------------------
// Private math helpers
// ---------------------------------------------------------------------------

function randUniform(a: number, b: number): number {
  return Math.random() * (b - a) + a;
}

function normalRandom(mean = 0, variance = 1): number {
  let v1: number, v2: number, s: number;
  do {
    v1 = 2 * Math.random() - 1;
    v2 = 2 * Math.random() - 1;
    s = v1 * v1 + v2 * v2;
  } while (s > 1);
  const result = Math.sqrt((-2 * Math.log(s)) / s) * v1;
  return mean + Math.sqrt(variance) * result;
}

function dist(a: Point2D, b: Point2D): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// utility
// ---------------------------------------------------------------------------

export function toOneHot(label: number, numClasses: number): number[]{
  const vec = new Array(numClasses).fill(0);
  vec[label] = 1;
  return vec;
}

export function discretizeLabels(
  examples: ExampleND[],
  numBins: number
): { examples: ExampleND[]; binLabels: string[] } {
  const values = examples.map(e => e.label);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const binSize = (max - min) / numBins;
 
  const allIntegers = values.every(v => Number.isInteger(v));
  const fmt = (v: number) => allIntegers ? String(Math.round(v)) : v.toFixed(1);
  const binLabels = Array.from({length: numBins}, (_, i) => {
    const lo = fmt(min + i * binSize);
    const hi = fmt(min + (i + 1) * binSize);
    return `[${lo}-${hi}]`;
  });
 
  const remapped = examples.map(e => ({
    ...e,
    label: Math.min(numBins - 1, Math.floor((e.label - min) / binSize))
  }));
 
  return { examples: remapped, binLabels };
}