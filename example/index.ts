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

async function ml(data, label){

  const svm = new SVM();
  svm.feedSamples(data, label);
  await svm.train();

 

  // const is_loaded = await svm.load("fme.txt");
  // console.log("load_model",is_loaded);  // check if model is loaded

  // const is_saved = await svm.save("newfme.txt"); //saved the model
  // console.log("saved_model",is_saved); // check if the model is saved

  // const pred2 = await svm.predict(data[7]);
  // console.log(`Prediction: ${pred2}`);
  
  let pred_data: Array<number> = [];

  for(let i in data){
     let pred = await svm.predict(data[i])
     pred_data.push(pred);
  }

  const evaluate = svm.evaluate(label, pred_data);
  console.log(evaluate)
}

ml(data,label)