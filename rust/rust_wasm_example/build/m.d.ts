/* tslint:disable */
/* eslint-disable */
/**
*/
export class ConvNetModel {
  free(): void;
/**
* @param {Uint8Array} weights
*/
  constructor(weights: Uint8Array);
/**
* @param {Uint8Array} image_data
* @param {number} width
* @param {number} height
* @returns {any}
*/
  predict_image(image_data: Uint8Array, width: number, height: number): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_convnetmodel_free: (a: number) => void;
  readonly convnetmodel_new: (a: number, b: number, c: number) => void;
  readonly convnetmodel_predict_image: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly main: (a: number, b: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {SyncInitInput} module
*
* @returns {InitOutput}
*/
export function initSync(module: SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {InitInput | Promise<InitInput>} module_or_path
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: InitInput | Promise<InitInput>): Promise<InitOutput>;
