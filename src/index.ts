import * as module from '../dist/libsvm';
let libsvm;

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
    svm_type ? : SVM_TYPE;
    kernel_type ? : KERNEL_TYPE;
    degree ? : number;
    gamma ? : number;
    coef0 ? : number;
    cache_size ? : number;
    C ? : number;
    nr_weight ? : number;
    weight_label ? : Array < number > ;
    weight ? : Array < number > ;
    nu ? : number;
    p ? : number;
    shrinking ? : number;
    probability ? : number;
}

class SVMParam {
    public svm_type? = SVM_TYPE.C_SVC;
    public kernel_type? = KERNEL_TYPE.RBF;
    public degree?: number = 3;
    public gamma?: number = 0;
    public coef0?: number = 0;
    public cache_size?: number = 100;
    public C?: number = 1;
    public nr_weight?: number = 0;
    public weight_label?: Array < number > = [];
    public weight?: Array < number > = [];
    public nu?: number = 0.5;
    public p?: number = 0.1;
    public eps?: number = 1e-3;
    public shrinking?: number = 0;
    public probability?: number = 0;

    constructor(param: ISVMParam) {
        if (typeof param !== 'object') return;
        for (const key of Object.keys(param)) {
            this[key] = param[key];
        }
        if (this.svm_type === SVM_TYPE.EPSILON_SVR || this.svm_type === SVM_TYPE.NU_SVR) {
            if (this.gamma === 0) this.gamma = .1;
        } else {
            if (this.gamma === 0) this.gamma = .5;
        }
    }

}

class SVM {
    readonly param: SVMParam;
    private paramPointer: number;
    private samplesPointer: number;
    private modelPointer: number;
    private ready: Promise<void>;

    constructor(param?: SVMParam) {
        this.ready = new Promise(async (resolve) => {
            libsvm = await module();
            resolve();
        });
        
        if (!param) this.param = new SVMParam({});
        else this.param = param;
        this.feedParam();
    }

    public async train() {
        await this.ready;
        if (this.paramPointer == null || this.samplesPointer == null) return;
        if (this.modelPointer == null) libsvm._free_model(this.modelPointer);
        this.modelPointer = libsvm._train_model(this.samplesPointer, this.paramPointer);
    }

    public async predict(data: Array<number>): Promise<number> {
        await this.ready;

        if (this.modelPointer == null) {
            console.error("Model should be trained first");
            return;
        }

        const dataPtr = libsvm._malloc(data.length * 8);
        libsvm.HEAPF64.set(data, dataPtr/8);

        return libsvm._predict_one(this.modelPointer, dataPtr, data.length) as number;
    }

    public async feedSamples(data: Array<Array<number>>, labels: Array<number>) {
        await this.ready;

        if (this.samplesPointer == null) libsvm._free_sample(this.samplesPointer);

        const encodeData = new Float64Array(data.reduce((prev, curr) => prev.concat(curr), []));
        
        const dataPtr = libsvm._malloc(encodeData.length * 8);
        libsvm.HEAPF64.set(encodeData, dataPtr/8);
        
        const labelPtr = libsvm._malloc(labels.length * 8);
        libsvm.HEAPF64.set(encodeData, labelPtr/8);
        
        this.samplesPointer = libsvm._make_samples(dataPtr, labelPtr, data.length, data[0].length);
    }

    public async feedParam() {
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
            eps,
            p,
            shrinking,
            probability,
            nr_weight,
            weight_label,
            weight
        } = this.param;

        if (this.paramPointer == null) libsvm._free(this.paramPointer);

        const weightLabelPtr = libsvm._malloc(nr_weight * 4);
        libsvm.HEAP32.set(new Int32Array(weight_label), weightLabelPtr/4);
        
        const weightlPtr = libsvm._malloc(nr_weight * 8);
        libsvm.HEAPF64.set(new Float64Array(weight_label), weightlPtr/8);
        
        this.paramPointer = libsvm._make_param(
            svm_type, kernel_type, degree,
            gamma, coef0, nu, cache_size, C, eps, p,
            shrinking, probability, nr_weight,
            weightLabelPtr, weightlPtr
        );
    }

    public async save(name: string): Promise<boolean> {
        await this.ready;

        if (this.modelPointer == null) {
            console.error("Model should be trained first");
            return;
        }

        var buffer = libsvm._malloc(name.length+1);

        // Write the string to memory
        libsvm.stringToUTF8(name, buffer,name.length+1);
        // const pred: boolean = libsvm._save_model(this.modelPointer, buffer);

        return libsvm._save_model(this.modelPointer, buffer) as boolean;
    }


}

export default SVM;