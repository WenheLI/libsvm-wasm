export interface LibsvmModlue extends EmscriptenModule {
    stringToUTF8: typeof stringToUTF8;
    _train_model(samplesPointer: number, paramPointer: number): number;
    _free_model(modelPointer: number): void;
    _init_node(dataPointer: number, size: number): number;
    _save_model(modelPointer: number, filenamePointer: number): Boolean;
    _load_model(filenamePointer: number): number;
    _make_samples(dataPointer: number, labelPointer: number, nb_feat: number, nb_dim: number): number;
    _make_param(svm_type: number, kernel_type: number,
                degree: number, gamma: number, coef0: number,
                nu: number, cache_size: number, eps: number, C: number,
                p: number, shrinking: number, probability: number,
                nr_weight: number, weightLabelPointer: number, 
                weightPointer: number): number;
   _free_sample(samplesPointer: number): void;
   _cross_valid_model(samplesPointer: number, paramPointer: number, kFold: number, targetPointer: number);
   _predict_one(modelPointer: number, dataPointer: number, size: number): number;
   _predict_one_with_prob(modelPointer: number, dataPointer: number, size: number, probEstimates: number): number;
}

export default function libsvmFactory(): Promise<LibsvmModlue>;
