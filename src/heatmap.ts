import { ExampleND } from "./dataset";
import * as d3 from "d3";

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

const NUM_SHADES = 30;

export class HeatMap {
  private settings: HeatMapSettings = { showAxes: false, noSvg: false };
  private xScale: d3.ScaleLinear<number, number>;
  private yScale: d3.ScaleLinear<number, number>;
  private numSamples: number;
  private color: d3.ScaleQuantize<string>;
  private canvas: d3.Selection<HTMLCanvasElement, unknown, null, undefined>;
  private svg?: d3.Selection<SVGGElement, unknown, null, undefined>;

  constructor(
    width: number,
    numSamples: number,
    xDomain: [number, number],
    yDomain: [number, number],
    container: d3.Selection<HTMLElement, unknown, null, undefined>,
    userSettings?: HeatMapSettings
  ) {
    this.numSamples = numSamples;
    const height = width;
    const padding = userSettings?.showAxes ? 20 : 0;
    if (userSettings) Object.assign(this.settings, userSettings);

    // d3 v7: d3.scaleLinear() (was d3.scale.linear() in v3)
    this.xScale = d3.scaleLinear().domain(xDomain).range([0, width - 2 * padding]);
    this.yScale = d3.scaleLinear().domain(yDomain).range([height - 2 * padding, 0]);

    const tmpScale = d3.scaleLinear<string>()
      .domain([0, 0.5, 1])
      .range(["#f59322", "#e8eaeb", "#0877bd"])
      .clamp(true);

    const colors = d3.range(0, 1 + 1e-9, 1 / NUM_SHADES).map(a => tmpScale(a));
    this.color = d3.scaleQuantize<string>().domain([-1, 1]).range(colors);

    const div = container.append("div")
      .style("width", `${width}px`)
      .style("height", `${height}px`)
      .style("position", "relative")
      .style("top", `-${padding}px`)
      .style("left", `-${padding}px`);

    this.canvas = div.append("canvas")
      .attr("width", numSamples)
      .attr("height", numSamples)
      .style("width", `${width - 2 * padding}px`)
      .style("height", `${height - 2 * padding}px`)
      .style("position", "absolute")
      .style("top", `${padding}px`)
      .style("left", `${padding}px`);

    if (!this.settings.noSvg) {
      this.svg = (div.append("svg") as d3.Selection<SVGSVGElement, unknown, null, undefined>)
        .attr("width", width)
        .attr("height", height)
        .style("position", "absolute")
        .style("left", "0")
        .style("top", "0")
        .append("g")
        .attr("transform", `translate(${padding},${padding})`);

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    // d3 v7: d3.axisBottom / d3.axisLeft (was d3.svg.axis().orient() in v3)
    if (this.settings.showAxes && this.svg) {
      this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(0,${height - 2 * padding})`)
        .call(d3.axisBottom(this.xScale));
      this.svg.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(this.yScale));
    }
  }

  updateTestPoints(points: ExampleND[]): void {
    if (this.settings.noSvg || !this.svg) throw Error("Can't add points since noSvg=true");
    this.updateCircles(this.svg.select("g.test"), points);
  }

  updatePoints(points: ExampleND[]): void {
    if (this.settings.noSvg || !this.svg) throw Error("Can't add points since noSvg=true");
    this.updateCircles(this.svg.select("g.train"), points);
  }

  updateBackground(data: number[][], discretize: boolean): void {
    const dx = data[0].length;
    const dy = data.length;
    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error("Matrix must be numSamples x numSamples");
    }
    const context = (this.canvas.node() as HTMLCanvasElement).getContext("2d")!;
    const image = context.createImageData(dx, dy);
    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        if (discretize) value = value >= 0 ? 1 : -1;
        const c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  private updateCircles(
    container: d3.Selection<SVGGElement, unknown, null, undefined>,
    points: ExampleND[]
  ) {
    const [xMin, xMax] = this.xScale.domain();
    const [yMin, yMax] = this.yScale.domain();

    // ExampleND: features[0] = x, features[1] = y
    const filtered = points.filter(p => {
      const px = p.features[0], py = p.features[1];
      return px >= xMin && px <= xMax && py >= yMin && py <= yMax;
    });

    const circles = container
      .selectAll<SVGCircleElement, ExampleND>("circle")
      .data(filtered);

    circles.enter()
      .append("circle")
      .attr("r", 3)
      .merge(circles)
      .attr("cx", d => this.xScale(d.features[0]))
      .attr("cy", d => this.yScale(d.features[1]))
      .style("fill", d => this.color(d.label));

    circles.exit().remove();
  }
}

export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) throw new Error("Matrix must be square");
  if (matrix.length % factor !== 0) throw new Error("Size must be divisible by factor");
  const result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      for (let k = 0; k < factor; k++)
        for (let l = 0; l < factor; l++)
          avg += matrix[i + k][j + l];
      result[i / factor][j / factor] = avg / (factor * factor);
    }
  }
  return result;
}
