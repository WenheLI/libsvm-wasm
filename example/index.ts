import SVM from '../src/index';
import { readFileSync } from 'fs';
import { join } from 'path';

const rawData = readFileSync(join(__dirname, 'data'))
              .toString().split(' \n').map((it) => 
                it.split(' ').map((it, idx) => {
                  if (idx == 0) return it;
                  else return it.split(':')[1];
                })
              );


const label = rawData.map(it => parseInt(it[0]));
const data = rawData.map(it => it.slice(1).map(it => parseFloat(it)));

const svm = new SVM();
svm.feedSamples(data, label);
svm.train();