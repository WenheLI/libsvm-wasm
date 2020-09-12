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
    public param: SVMParam;
    public paramPointer: number;
    public samplesPointer: number;
    public modelPointer: number;
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
        this.modelPointer = libsvm._train_model(this.samplesPointer, this.paramPointer);
    }

    public async predict(data: Array<number>): Promise<number> {
        await this.ready;
        return libsvm._predict_one(this.modelPointer, data, data.length) as number;
    }

    public async feedSamples(data: Array<Array<number>>, labels: Array<number>) {
        await this.ready;
        const encodeData: Array<number> = data.reduce((prev, curr) => prev.concat(curr), []);
        this.samplesPointer = libsvm._make_samples(encodeData, labels, data.length, data[0].length);
    }

    private async feedParam() {
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
        this.paramPointer = libsvm._make_param(
            svm_type, kernel_type, degree,
            gamma, coef0, nu, cache_size, C, eps, p,
            shrinking, probability, nr_weight,
            weight_label, weight
        );
    }


}
const main = async () => {
    const features = [[0, 0], [1, 1], [1, 0], [0, 1]];
    const labels = [0, 0, 0, 0];
    
    const svm = new SVM({
        kernel_type: KERNEL_TYPE.RBF,
        svm_type: SVM_TYPE.ONE_CLASS,
        gamma: 1,
        nu: 0.1
    });
    console.log(1)
    
    await svm.feedSamples(features, labels);
    console.log(2)
    
    await svm.train();
    console.log(3)
    
    const toPredict = [[0.5, 0.5], [1.5, 1]];
    const expected = [1, -1];
    svm.predict(toPredict[0]).then((it) => console.log(it))
}

main()