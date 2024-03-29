# libsvm-wasm

## Brief
This is a wasm export for [`libsvm`](https://github.com/cjlin1/libsvm)
It allows you to directly run & train svm model in js env without installing any non-js dependency.

## Build

- prereq:
  [emscripten](https://emscripten.org/docs/getting_started/index.html)

```bash
git submodule init
git submodule update
make
```

After this command, you will get the `wasm` in dist folder.

## Usage

**Note:** No bundle support for now.
You may want to build on yourself or use `ts-node` to write a ts script.

```js
import {SVM} from './src/libsvm';

const svm = new SVM();
await svm.init()
const data = [[-1, -1], [1, 1], [2, 2], [-2, -2]]
const label = [-1, 1, 1, -1];

svm.feedSamples(data, label);
await svm.train();
svm.predict([3, 3]);
```

## Roadmap

- Building & Bundling & Packaging
- Benchmark V.S. native
- More API
- SIMD operation to enhance performance
- Store Model using native fs API
