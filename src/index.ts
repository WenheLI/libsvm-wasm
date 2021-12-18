import svmFactory, { LibsvmModlue } from '../dist/libsvm';
let libsvm: LibsvmModlue;

enum SVM_TYPE {
    C_SVC,
    NU_SVC,
    ONE_CLASS,
    EPSILON_SVR,
    NU_SVR
}

enum KERNEL_TYPE {
    LINEAR,
    POLY,
    RBF,
    SIGMOID,
    PRECOMPUTED
}

interface ISVMParam {
    svm_type: SVM_TYPE;
    kernel_type: KERNEL_TYPE;
    degree: number;
    gamma: number;
    coef0: number;
    cache_size: number;
    C: number;
    nr_weight: number;
    weight_label: Array<number> ;
    weight: Array<number> ;
    nu: number;
    p: number;
    shrinking: number;
    probability: number;
}

class SVMParam {

    public param: ISVMParam = {
        svm_type: SVM_TYPE.C_SVC,
        kernel_type: KERNEL_TYPE.RBF,
        degree: 3,
        gamma: 0,
        coef0: 0,
        cache_size: 100,
        C: 1,
        nr_weight: 0,
        weight_label: [],
        weight: [],
        nu: 0.5,
        p: 0.1,
        shrinking: 0,
        probability: 0
    };

    public eps: number = 1e-3;

    constructor(param: ISVMParam, eps: number = 1e-3) {
        if (typeof param !== 'object') return;
        this.param = {
            ...this.param,
            ...param
        };

        this.eps = eps;

        if (this.param.svm_type === SVM_TYPE.EPSILON_SVR || this.param.svm_type === SVM_TYPE.NU_SVR) {
            if (this.param.gamma === 0) this.param.gamma = .1;
        } else {
            if (this.param.gamma === 0) this.param.gamma = .5;
        }
    }

}

class SVM {
    readonly param: SVMParam;
    private paramPointer: number = -1;
    private samplesPointer: number = -1;
    private modelPointer: number = -1;
    private ready: Promise<void>;

    constructor(param?: SVMParam) {
        this.ready = new Promise((resolve, reject) => {
            svmFactory().then((lib) => {
                    libsvm = lib;
                    resolve();
                });
        });
        if (!param) this.param = new SVMParam({} as ISVMParam);
        else this.param = param;
    }

    public train = async () => {
        await this.ready;
        if (this.paramPointer == -1) {
            await this.feedParam();
        }
        if (this.paramPointer == -1 || this.samplesPointer == -1) return;
        if (this.modelPointer != -1) libsvm._free_model(this.modelPointer);
        this.modelPointer = libsvm._train_model(this.samplesPointer, this.paramPointer);
    }

    public predict = async (data: Array<number>): Promise<number> => {
        await this.ready;
        if (this.modelPointer == null) {
            console.error("Model should be trained first");
            return -1;
        }

        const dataPtr = libsvm._malloc(data.length * 8);
        libsvm.HEAPF64.set(data, dataPtr/8);

        return libsvm._predict_one(this.modelPointer, dataPtr, data.length) as number;
    }

    public feedSamples = async (data: Array<Array<number>>, labels: Array<number>) => {
        await this.ready;
        if (this.samplesPointer == null) libsvm._free_sample(this.samplesPointer);

        const encodeData = new Float64Array(data.reduce((prev, curr) => prev.concat(curr), []));
        
        const dataPtr = libsvm._malloc(encodeData.length * 8);
        libsvm.HEAPF64.set(encodeData, dataPtr/8);
        
        const labelPtr = libsvm._malloc(labels.length * 8);
        libsvm.HEAPF64.set(encodeData, labelPtr/8);
        
        this.samplesPointer = libsvm._make_samples(dataPtr, labelPtr, data.length, data[0].length);
    }

    public feedParam = async () => {
        await this.ready;
        const {
            svm_type,
            kernel_type,
            degree,
            gamma,
            coef0,
            nu,
            cache_size,
            C,
            p,
            shrinking,
            probability,
            nr_weight,
            weight_label,
            weight
        } = this.param.param;
        const eps = this.param.eps;

        if (this.paramPointer == null) libsvm._free(this.paramPointer);

        const weightLabelPtr = libsvm._malloc(nr_weight * 4);
        libsvm.HEAP32.set(new Int32Array(weight_label), weightLabelPtr/4);
        
        const weightlPtr = libsvm._malloc(nr_weight * 8);
        libsvm.HEAPF64.set(new Float64Array(weight), weightlPtr/8);
        
        this.paramPointer = libsvm._make_param(
            svm_type, kernel_type, degree,
            gamma, coef0, nu, cache_size, C, eps, p,
            shrinking, probability, nr_weight,
            weightLabelPtr, weightlPtr
        );
    }

    public save = async (name: string): Promise<boolean> => {
        await this.ready;
        if (this.modelPointer == null) {
            console.error("Model should be trained first");
            return false;
        }

        const buffer = libsvm._malloc(name.length+1);

        // Write the string to memory
        libsvm.stringToUTF8(name, buffer,name.length+1);

        return libsvm._save_model(this.modelPointer, buffer) as boolean;
    }

    public load = async (name: string): Promise<boolean> => {
        await this.ready;
        if (this.modelPointer == null) libsvm._free_model(this.modelPointer);

        const buffer = libsvm._malloc(name.length+1);
        libsvm.stringToUTF8(name, buffer,name.length+1);

        this.modelPointer = libsvm._load_model(buffer);

        if(this.modelPointer != null || this.modelPointer > 0){
            return true;
        }
        return false;

    }
    public evaluate = (label: Array<number>, pred: Array<number>): Array<string> =>{
        //from the python implementation libsvm/python/commonutil.py
        if(label.length != pred.length){
            throw new Error(`length mismatch; actual label ${label.length} does not match prediction of length ${pred.length}`);
        }

        let total_correct: number = 0;
        let total_error: number = 0;
        let sumv = 0;
        let sumy = 0;
        let sumvv = 0;
        let sumyy = 0;
        let sumvy = 0;
        
        for(let i=0; i < label.length; i++){
            let label_i = label[i];
            let pred_i  = pred[i];

            if(label_i === pred_i){
                total_correct +=1;
            }

            total_error += (pred_i-label_i)*(pred_i-label_i);
            sumv += pred_i;
            sumy += label_i;
            sumvv += pred_i*pred_i;
            sumyy += label_i*label_i;
            sumvy += pred_i * label_i

        }
        let len = label.length;
        let ACC = 100.0*total_correct/len;
        let MSE = total_error/len;
        let SCC: number;
        try{
            SCC = ((len*sumvy-sumv*sumy)*(len*sumvy-sumv*sumy))/((len*sumvv-sumv*sumv)*(len*sumyy-sumy*sumy));
        }
        catch(e){
            SCC = parseFloat("nan");
        }
        return [ACC.toFixed(2),MSE.toFixed(3),SCC.toFixed(3)];
    }


}

export {
    SVM,
    SVM_TYPE,
    KERNEL_TYPE,
    SVMParam
};