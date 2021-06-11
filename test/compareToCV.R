library(agghoo)

standardCV <- function(data, target, task = NULL, gmodel = NULL, params = NULL,
  loss = NULL, CV = list(type = "MC", V = 10, test_size = 0.2, shuffle = TRUE)
) {
  if (!is.null(task))
    task = match.arg(task, c("classification", "regression"))
  if (is.character(gmodel))
    gmodel <- match.arg(gmodel, c("knn", "ppr", "rf", "tree"))
  if (is.numeric(params) || is.character(params))
    params <- as.list(params)
  if (is.null(task)) {
    if (is.numeric(target))
      task = "regression"
    else
      task = "classification"
  }

  if (is.null(loss)) {
    loss <- function(y1, y2) {
      if (task == "classification") {
        if (is.null(dim(y1)))
          mean(y1 != y2)
        else {
          if (!is.null(dim(y2)))
            mean(rowSums(abs(y1 - y2)))
          else {
            y2 <- as.character(y2)
            names <- colnames(y1)
            positions <- list()
            for (idx in seq_along(names))
              positions[[ names[idx] ]] <- idx
            mean(vapply(
              seq_along(y2),
              function(idx) sum(abs(y1[idx,] - positions[[ y2[idx] ]])),
              0))
          }
        }
      }
      else
        mean(abs(y1 - y2))
    }
  }

  n <- nrow(data)
  shuffle_inds <- NULL
  if (CV$type == "vfold" && CV$shuffle)
    shuffle_inds <- sample(n, n)
  get_testIndices <- function(v, shuffle_inds) {
    if (CV$type == "vfold") {
      first_index = round((v-1) * n / CV$V) + 1
      last_index = round(v * n / CV$V)
      test_indices = first_index:last_index
      if (!is.null(shuffle_inds))
        test_indices <- shuffle_inds[test_indices]
    }
    else
      test_indices = sample(n, round(n * CV$test_size))
    test_indices
  }
  list_testinds <- list()
  for (v in seq_len(CV$V))
    list_testinds[[v]] <- get_testIndices(v, shuffle_inds)

  gmodel <- agghoo::Model$new(data, target, task, gmodel, params)
  best_error <- Inf
  best_model <- NULL
  for (p in seq_len(gmodel$nmodels)) {
    error <- 0
    for (v in seq_len(CV$V)) {
      testIdx <- list_testinds[[v]]
      dataHO <- data[-testIdx,]
      testX <- data[testIdx,]
      targetHO <- target[-testIdx]
      testY <- target[testIdx]
      if (!is.matrix(dataHO) && !is.data.frame(dataHO))
        dataHO <- as.matrix(dataHO)
      if (!is.matrix(testX) && !is.data.frame(testX))
        testX <- as.matrix(testX)
      model_pred <- gmodel$get(dataHO, targetHO, p)
      prediction <- model_pred(testX)
      error <- error + loss(prediction, testY)
    }
    if (error <= best_error) {
      newModel <- list(model=model_pred, param=gmodel$getParam(p))
      if (error == best_error)
        best_model[[length(best_model)+1]] <- newModel
      else {
        best_model <- list(newModel)
        best_error <- error
      }
    }
  }
  best_model[[ sample(length(best_model), 1) ]]
}

compareToCV <- function(df, t_idx, task=NULL, rseed=-1, verbose=TRUE, ...) {
  if (rseed >= 0)
    set.seed(rseed)
  if (is.null(task))
    task <- ifelse(is.numeric(df[,t_idx]), "regression", "classification")
  n <- nrow(df)
  test_indices <- sample( n, round(n / ifelse(n >= 500, 10, 5)) )
  a <- agghoo(df[-test_indices,-t_idx], df[-test_indices,t_idx], task, ...)
  a$fit()
  if (verbose) {
    print("Parameters:")
    print(unlist(a$getParams()))
  }
  pa <- a$predict(df[test_indices,-t_idx])
  err_a <- ifelse(task == "classification",
                  mean(pa != df[test_indices,t_idx]),
                  mean(abs(pa - df[test_indices,t_idx])))
  if (verbose)
    print(paste("error agghoo:", err_a))
  # Compare with standard cross-validation:
  s <- standardCV(df[-test_indices,-t_idx], df[-test_indices,t_idx], task, ...)
  if (verbose)
    print(paste( "Parameter:", s$param ))
  ps <- s$model(df[test_indices,-t_idx])
  err_s <- ifelse(task == "classification",
                  mean(ps != df[test_indices,t_idx]),
                  mean(abs(ps - df[test_indices,t_idx])))
  if (verbose)
    print(paste("error CV:", err_s))
  invisible(c(err_a, err_s))
}

library(parallel)
compareMulti <- function(df, t_idx, task = NULL, N = 100, nc = NA, ...) {
  if (is.na(nc))
    nc <- detectCores()
  errors <- mclapply(1:N,
                     function(n) {
                       compareToCV(df, t_idx, task, n, verbose=FALSE, ...) },
                     mc.cores = nc)
  print("error agghoo vs. cross-validation:")
  Reduce('+', errors) / N
}
