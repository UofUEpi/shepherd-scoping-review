Model
================

## Quarto

Quarto enables you to weave together content and executable code into a
finished document. To learn more about Quarto see <https://quarto.org>.

## Reading the data in

We use the R packages tensorflow and readxl

    Loading required package: NLP

    AUC 0.3.2

    Type AUCNews() to see the change log and ?AUC to get an overview.

## Identifying key concepts

                             V1  N
        1:         copyright â© 91
        2:          of covid-19 65
        3:         the covid-19 51
        4:            number of 45
        5:           this study 41
       ---                        
    25346:   zero exponentially  1
    25347:              zero in  1
    25348: zhang, dhanalakshmi,  1
    25349:           zone, such  1
    25350:         zubair shah.  1

# Generating the data matrix

``` r
data_mat <- matrix(
  0L,
  nrow = n, ncol = k, 
  dimnames = list(1:n, keyterms)
  )

# One term
idx_1 <- Map(\(a,b) {
  a <- intersect(a, keyterms)
  cbind(rep(b, length(a)), a)
  }, a = ngrams_1, b = 1:n) |>
  do.call(what = rbind)

data_mat[idx_1] <- 1L

# Two terms
idx_1 <- Map(\(a,b) {
  a <- intersect(a, keyterms)
  cbind(rep(b, length(a)), a)
  }, a = ngrams_2, b = 1:n) |>
  do.call(what = rbind)

data_mat[idx_1] <- 1L

# Three terms
idx_1 <- Map(\(a,b) {
  a <- intersect(a, keyterms)
  cbind(rep(b, length(a)), a)
  }, a = ngrams_3, b = 1:n) |>
  do.call(what = rbind)

data_mat[idx_1] <- 1L

# Four terms
idx_1 <- Map(\(a,b) {
  a <- intersect(a, keyterms)
  cbind(rep(b, length(a)), a)
  }, a = ngrams_4, b = 1:n) |>
  do.call(what = rbind)

data_mat[idx_1] <- 1L
```

``` r
library(Matrix)
as(data_mat, "dgCMatrix") |> image(main = "Model matrix\n(top keywords, 2-grams, 3-grams, and 4-grams)")
```

![](model0_files/figure-gfm/Data%20viz-1.png)

# Building the convolutional neural network (data)

``` r
set.seed(1231)
n_yes <- nrow(data_yes)
n_no  <- nrow(data_no)

prop_train <- .7

train_yes_id <- sample.int(n_yes, n_yes * prop_train) |> sort()
train_no_id  <- sample.int(n_no, n_no * prop_train) |> sort()

n_train <- length(train_yes_id) + length(train_no_id)

# Features
x_train <- data_mat[c(train_yes_id, n_yes + train_no_id),] |>
  as.vector() |>
  array_reshape(c(n_train, k), order = "F")

x_test <- data_mat[c(-train_yes_id, -(n_yes + train_no_id)),] |>
  as.vector() |>
  array_reshape(c(n - n_train, k), order = "F")

# Labels
y_train <- c(rep(1, length(train_yes_id)), rep(0, length(train_no_id))) |>
  array_reshape(c(n_train, 1), order = "F")

y_test <- c(rep(1, n_yes - length(train_yes_id)), rep(0, n_no - length(train_no_id))) |>
  array_reshape(c(n - n_train, 1), order = "F")
```

# Building the model

``` r
# Setting the seed
library(keras)
tensorflow::set_random_seed(5554)
```

    Loaded Tensorflow version 2.9.1

``` r
# Building the model
model <- keras_model_sequential(
  name = "LitRev"
) |>
  # Entry layer
  layer_dense(
    units = (k), input_shape = k
    ) |>
  layer_dropout(rate = .2) |>
  layer_activation_relu() |>
  layer_dense(units = 1, activation = "sigmoid")
  

model |> compile(
  loss = loss_mean_absolute_error,
  keras::optimizer_adam(learning_rate = .01),
  metrics = c("mae"), weighted_metrics = list(NULL), 
)

# Fitting the model
history <- model |>
  fit(
    x_train, y_train,
    epochs = 150,
    batch_size = 16 * 4, 
    verbose = 2,
    validation_split = .2 #, sample_weight = w
    )

# plot(history)
# 
# # Guardando modelo -------------------------------------------------------------
# if (!dir.exists("modelo-00-keras")) {
#   dir.create("modelo-00-keras")
# } else {
#   file.remove(list.files("modelo-00-keras", all.files = TRUE, full.names = TRUE))
# }
# 
# model$save("modelo-00-keras")
```

``` r
evaluate(model, x_test, y_test)
```

        loss      mae 
    0.274997 0.274997 

``` r
p <- predict(model, x = x_test)

aucs <- AUC::roc(
  p, labels = as.factor(y_test)
) 
plot(aucs, main = sprintf("AUC-ROC %.2f", auc(aucs)))
```

![](model0_files/figure-gfm/Evaluation-1.png)
