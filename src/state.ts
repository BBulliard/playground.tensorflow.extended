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
import * as dataset from "./dataset";

/** Suffix added to the state when storing if a control is hidden or not. */
const HIDE_STATE_SUFFIX = "_hide";

/** A map between names and activation functions. */
export let activations: {[key: string]: nn.ActivationFunction} = {
  "relu": nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  "sigmoid": nn.Activations.SIGMOID,
  "linear": nn.Activations.LINEAR
};

/** A map between names and regularization functions. */
export let regularizations: {[key: string]: nn.RegularizationFunction | null} = {
  "none": null,
  "L1": nn.RegularizationFunction.L1,
  "L2": nn.RegularizationFunction.L2
};

/** A map between dataset names and functions that generate classification data. */
export let datasets: {[key: string]: dataset.DataGenerator} = {
  "circle": dataset.classifyCircleData,
  "xor": dataset.classifyXORData,
  "simple": dataset.classifySimpleData,
  "spiral": dataset.classifySpiralData,
  "checkerboard": dataset.classifyCheckerboardData,
  "two-moons": dataset.classifyTwoMoonsData,
  "diagonal-stripes": dataset.classifyDiagonalStripesData
};

/** A map between dataset names and functions that generate regression data. */
export let regDatasets: {[key: string]: dataset.DataGenerator} = {
  "reg-plane": dataset.regressPlane,
  "reg-gauss": dataset.regressGaussian
};

export function getKeyFromValue(obj: any, value: any): string | undefined {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

function endsWith(s: string, suffix: string): boolean {
  return s.substr(-suffix.length) === suffix;
}

function getHideProps(obj: any): string[] {
  let result: string[] = [];
  for (let prop in obj) {
    if (endsWith(prop, HIDE_STATE_SUFFIX)) {
      result.push(prop);
    }
  }
  return result;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: {[key: string]: any};
};

// ---------------------------------------------------------------------------
// State class
// ---------------------------------------------------------------------------

export class State {

  private static PROPS: Property[] = [
    {name: "activation", type: Type.OBJECT, keyMap: activations},
    {name: "regularization", type: Type.OBJECT, keyMap: regularizations},
    {name: "batchSize", type: Type.NUMBER},
    {name: "dataset", type: Type.OBJECT, keyMap: datasets},
    {name: "regDataset", type: Type.OBJECT, keyMap: regDatasets},
    {name: "learningRate", type: Type.NUMBER},
    {name: "regularizationRate", type: Type.NUMBER},
    {name: "noise", type: Type.NUMBER},
    {name: "networkShape", type: Type.ARRAY_NUMBER},
    {name: "seed", type: Type.STRING},
    {name: "showTestData", type: Type.BOOLEAN},
    {name: "discretize", type: Type.BOOLEAN},
    {name: "percTrainData", type: Type.NUMBER},
    // Legacy 2-D input toggles – kept for URL backward-compatibility.
    // They are used only when activeInputs is empty (no ND dataset loaded).
    {name: "x", type: Type.BOOLEAN},
    {name: "y", type: Type.BOOLEAN},
    {name: "xTimesY", type: Type.BOOLEAN},
    {name: "xSquared", type: Type.BOOLEAN},
    {name: "ySquared", type: Type.BOOLEAN},
    {name: "cosX", type: Type.BOOLEAN},
    {name: "sinX", type: Type.BOOLEAN},
    {name: "cosY", type: Type.BOOLEAN},
    {name: "sinY", type: Type.BOOLEAN},
    {name: "collectStats", type: Type.BOOLEAN},
    {name: "tutorial", type: Type.STRING},
    {name: "problem", type: Type.OBJECT, keyMap: problems},
    {name: "initZero", type: Type.BOOLEAN},
    {name: "hideText", type: Type.BOOLEAN},
    // ND extension: comma-separated list of active feature names.
    // When non-empty, completely overrides the legacy x/y/... toggles.
    {name: "activeInputs", type: Type.ARRAY_STRING},
    // Whether the current data comes from a user-uploaded CSV file.
    {name: "isCSVDataset", type: Type.BOOLEAN},
  ];

  [key: string]: any;

  // ---- Hyperparameters (unchanged) ----------------------------------------
  learningRate = 0.03;
  regularizationRate = 0;
  showTestData = false;
  noise = 0;
  batchSize = 10;
  discretize = false;
  tutorial: string | null = null;
  percTrainData = 50;
  activation = nn.Activations.TANH;
  regularization: nn.RegularizationFunction | null = null;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  hideText = false;
  collectStats = false;
  numHiddenLayers = 1;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [4, 2];

  // ---- Legacy 2-D input toggles -------------------------------------------
  x = true;
  y = true;
  xTimesY = false;
  xSquared = false;
  ySquared = false;
  cosX = false;
  sinX = false;
  cosY = false;
  sinY = false;

  // ---- Dataset references -------------------------------------------------
  dataset: dataset.DataGenerator = dataset.classifyCircleData;
  regDataset: dataset.DataGenerator = dataset.regressPlane;
  seed: string = "";

  // ---- ND extensions -------------------------------------------------------

  /**
   * When non-empty, this list defines exactly which input nodes are active
   * (by feature name). It completely replaces the legacy x/y/... boolean
   * toggles. Populated automatically when an ND dataset is loaded (either a
   * CSV file or an ND built-in generator).
   */
  activeInputs: string[] = [];

  /**
   * this lists defines which inputs are actually used
   */
  selectedInputs: string[] = [];

  /**
   * The raw CSV examples cached in memory so that train/test splits can be
   * recomputed without re-parsing. Never serialised.
   */
  csvData: dataset.ExampleND[] | null = null;

  /**
   * Feature names reported by the currently active ND dataset.
   * Derived from the first example's featureNames field; rebuilt whenever
   * data is (re-)generated. Never serialised.
   */
  currentFeatureNames: string[] = [];

  /**
   * column chosen as a label
   */
  csvLabelColumn: string | null = null;

  /**
   * all columns from csv
   */
  csvAllColumns: string[] = [];

  /**
   * raw text from csv in order to re-parse it 
   */
  csvRawText: string | null = null;

  /**
   * number of output classes
   * 1 by default
   */
  numClasses: number = 1;

  classLabels: string[] = [];

  // ---------------------------------------------------------------------------
  // Deserialize
  // ---------------------------------------------------------------------------

  static deserializeState(): State {
    let map: {[key: string]: string} = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    let state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    State.PROPS.forEach(({name, type, keyMap}) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
                "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    getHideProps(map).forEach(prop => {
      state[prop] = (map[prop] === "true") ? true : false;
    });

    state.numHiddenLayers = state.networkShape.length;
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    Math.seedrandom(state.seed);
    return state;
  }

  // ---------------------------------------------------------------------------
  // Serialize
  // ---------------------------------------------------------------------------

  serialize() {
    let props: string[] = [];
    State.PROPS.forEach(({name, type, keyMap}) => {
      // Never persist transient CSV fields to the URL.
      if (name === "isCSVDataset") return;

      let value = this[name];
      if (value == null) return;

      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER || type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${value}`);
    });

    getHideProps(this).forEach(prop => {
      props.push(`${prop}=${this[prop]}`);
    });

    window.location.hash = props.join("&");
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  getHiddenProps(): string[] {
    let result: string[] = [];
    for (let prop in this) {
      if (endsWith(prop, HIDE_STATE_SUFFIX) && String(this[prop]) === "true") {
        result.push(prop.replace(HIDE_STATE_SUFFIX, ""));
      }
    }
    return result;
  }

  setHideProperty(name: string, hidden: boolean) {
    this[name + HIDE_STATE_SUFFIX] = hidden;
  }
}
