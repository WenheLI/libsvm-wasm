#include "libsvm/svm.h"
#include <stdlib.h>
#include <stdio.h>
#include "emscripten.h"

#define true 1
#define false 0
typedef short bool;

#define MALLOC(type, size) (type *)malloc((size) * sizeof(type))

typedef struct svm_node svm_node;
typedef struct svm_model svm_model;
typedef struct svm_problem svm_problem;
typedef struct svm_parameter svm_parameter;

void print_null(const char *s) {}

EMSCRIPTEN_KEEPALIVE
svm_node *init_node(double *data, int size)
{
    svm_node *node = MALLOC(svm_node, size + 1);
    for (int i = 0; i < size; i++)
    {
        node[i].index = i + 1;
        node[i].value = data[i];
    }
    node[size].index = -1;
    return node;
}

EMSCRIPTEN_KEEPALIVE
bool save_model(svm_model *model, const char *model_file_name)
{
    int success = svm_save_model(model, model_file_name);
    if (success < 0)
        return false;
    return true;
}

// EMSCRIPTEN_KEEPALIVE
// svm_model* load_model(char* model) {
//     FILE* f = fopen("model.txt", "w");
//     fprintf(f, "%s", model);
//     fcloase(f);
//     return svm_load_model("model.txt");
// }

EMSCRIPTEN_KEEPALIVE
void free_model(svm_model *model)
{
    svm_free_and_destroy_model(&model);
}

svm_problem *make_samples_internal(int nb_feat, int nb_dim)
{
    svm_problem *sample = MALLOC(svm_problem, 1);
    sample->l = nb_feat;
    sample->y = MALLOC(double, nb_feat);
    sample->x = MALLOC(svm_node *, nb_feat);
    svm_node *space = MALLOC(svm_node, nb_feat * (nb_dim + 1));
    for (int i = 0; i < nb_feat; i++)
    {
        sample->x[i] = space + i * (nb_dim + 1);
    }
    return sample;
}

EMSCRIPTEN_KEEPALIVE
svm_problem *make_samples(double *data, double *labels, int nb_feat, int nb_dim)
{
    svm_problem *samples = make_samples_internal(nb_feat, nb_dim);
    for (int i = 0; i < nb_feat; i++)
    {
        for (int j = 0; j < nb_dim; j++)
        {
            samples->x[i][j].index = j + 1;
            samples->x[i][j].value = data[i * nb_dim + j];
        }
        samples->x[i][nb_dim].index = -1;
        samples->y[i] = labels[i];
    }
    return samples;
}

EMSCRIPTEN_KEEPALIVE
svm_parameter *make_param(
    int svm_type, int kernel_type, int degree, float gamma,
    float coef0, float nu, float cache_size, float C, float eps,
    float p, int shrinking, int probability, int nr_weight,
    int *weight_label, float *weight)
{
    svm_parameter *param = MALLOC(svm_parameter, 1);
    param->svm_type = svm_type;
    param->kernel_type = kernel_type;
    param->degree = degree;
    param->gamma = gamma;
    param->coef0 = coef0;
    param->nu = nu;
    param->cache_size = cache_size;
    param->C = C;
    param->eps = eps;
    param->p = p;
    param->shrinking = shrinking;
    param->probability = probability;
    param->nr_weight = nr_weight;
    param->weight_label = weight_label;
    param->weight = weight;
    return param;
}

EMSCRIPTEN_KEEPALIVE
svm_model *train_model(svm_problem *samples, svm_parameter *param)
{
    // svm_set_print_string_function(printf);
    svm_model *model = svm_train(samples, param);
    svm_destroy_param(param);
    return model;
}

EMSCRIPTEN_KEEPALIVE
void free_sample(svm_problem *samples)
{
    free(samples->y);
    free(samples->x);
    free(samples);
}

EMSCRIPTEN_KEEPALIVE
void cross_valid_model(svm_problem *samples, svm_parameter *param, int kFold, double *target)
{
    svm_cross_validation(samples, param, kFold, target);
    svm_destroy_param(param);
}

EMSCRIPTEN_KEEPALIVE
double predict_one(svm_model *model, double *data, int size)
{
    svm_node *node = init_node(data, size);
    double pred = svm_predict(model, node);
    free(node);
    return pred;
}

EMSCRIPTEN_KEEPALIVE
double predict_one_with_prob(svm_model *model, double *data, int size, double *prob_estimates)
{
    svm_node *node = init_node(data, size);
    double pred = svm_predict_probability(model, node, prob_estimates);
    free(node);
    return pred;
}
