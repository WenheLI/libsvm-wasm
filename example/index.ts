import SVM from '../src/';
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

async function ml(data: any, label: any){

  const svm = new SVM();
  svm.feedSamples(data, label);
  await svm.train();
  
  let pred_data: Array<number> = [];

  for(let i in data){
     let pred = await svm.predict(data[i])
     pred_data.push(pred);
  }

  const evaluate = svm.evaluate(label, pred_data);
  console.log(evaluate)
}

ml(data,label)