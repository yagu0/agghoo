#' @title R6 class with agghoo functions fit() and predict().
#'
#' @description
#' Class encapsulating the methods to run to obtain the best predictor
#' from the list of models (see 'Model' class).
#'
#' @importFrom R6 R6Class
#'
#' @export
AgghooCV <- R6::R6Class("AgghooCV",
  public = list(
    #' @description Create a new AgghooCV object.
    #' @param data Matrix or data.frame
    #' @param target Vector of targets (generally numeric or factor)
    #' @param task "regression" or "classification"
    #' @param gmodel Generic model returning a predictive function
    #' @param quality Function assessing the quality of a prediction;
    #'                quality(y1, y2) --> real number
    initialize = function(data, target, task, gmodel, quality = NULL) {
      private$data <- data
      private$target <- target
      private$task <- task
      private$gmodel <- gmodel
      if (is.null(quality)) {
        quality <- function(y1, y2) {
          # NOTE: if classif output is a probability matrix, adapt.
          if (task == "classification")
            mean(y1 == y2)
          else
            atan(1.0 / (mean(abs(y1 - y2) + 0.01))) #experimental...
        }
      }
      private$quality <- quality
    },
    #' @description Fit an agghoo model.
    #' @param CV List describing cross-validation to run. Slots:
    #'          - type: 'vfold' or 'MC' for Monte-Carlo (default: MC)
    #'          - V: number of runs (default: 10)
    #'          - test_size: percentage of data in the test dataset, for MC
    #'            (irrelevant for V-fold). Default: 0.2.
    #'          - shuffle: wether or not to shuffle data before V-fold.
    #'            Irrelevant for Monte-Carlo; default: TRUE
    #' @param mode "agghoo" or "standard" (for usual cross-validation)
    fit = function(
      CV = list(type = "MC",
                V = 10,
                test_size = 0.2,
                shuffle = TRUE),
      mode="agghoo"
    ) {
      if (!is.list(CV))
        stop("CV: list of type, V, [test_size], [shuffle]")
      n <- nrow(private$data)
      shuffle_inds <- NA
      if (CV$type == "vfold" && CV$shuffle)
        shuffle_inds <- sample(n, n)
      if (mode == "agghoo") {
        vperfs <- list()
        for (v in 1:CV$V) {
          test_indices <- private$get_testIndices(CV, v, n, shuffle_inds)
          vperf <- private$get_modelPerf(test_indices)
          vperfs[[v]] <- vperf
        }
        private$run_res <- vperfs
      }
      else {
        # Standard cross-validation
        best_index = 0
        best_perf <- -1
        for (p in 1:private$gmodel$nmodels) {
          tot_perf <- 0
          for (v in 1:CV$V) {
            test_indices <- private$get_testIndices(CV, v, n, shuffle_inds)
            perf <- private$get_modelPerf(test_indices, p)
            tot_perf <- tot_perf + perf / CV$V
          }
          if (tot_perf > best_perf) {
            # TODO: if ex-aequos: models list + choose at random
            best_index <- p
            best_perf <- tot_perf
          }
        }
        best_model <- private$gmodel$get(private$data, private$target, best_index)
        private$run_res <- list( list(model=best_model, perf=best_perf) )
      }
    },
    #' @description Predict an agghoo model (after calling fit())
    #' @param X Matrix or data.frame to predict
    #' @param weight "uniform" (default) or "quality" to weight votes or
    #'               average models performances (TODO: bad idea?!)
    predict = function(X, weight="uniform") {
      if (!is.matrix(X) && !is.data.frame(X))
        stop("X: matrix or data.frame")
      if (!is.list(private$run_res)) {
        print("Please call $fit() method first")
        return
      }
      V <- length(private$run_res)
      if (V == 1)
        # Standard CV:
        return (private$run_res[[1]]$model(X))
      # Agghoo:
      if (weight == "uniform")
        weights <- rep(1 / V, V)
      else {
        perfs <- sapply(private$run_res, function(item) item$perf)
        perfs[perfs < 0] <- 0 #TODO: show a warning (with count of < 0...)
        total_weight <- sum(perfs) #TODO: error if total_weight == 0
        weights <- perfs / total_weight
      }
      n <- nrow(X)
      # TODO: detect if output = probs matrix for classif (in this case, adapt?)
      # prediction agghoo "probabiliste" pour un nouveau x :
      # argMax({ predict(m_v, x), v in 1..V }) ...
      if (private$task == "classification") {
        votes <- as.list(rep(NA, n))
        parse_numeric <- FALSE
      }
      else
        preds <- matrix(0, nrow=n, ncol=V)
      for (v in 1:V) {
        predictions <- private$run_res[[v]]$model(X)
        if (private$task == "regression")
          preds <- cbind(preds, weights[v] * predictions)
        else {
          if (!parse_numeric && is.numeric(predictions))
            parse_numeric <- TRUE
          for (i in 1:n) {
            if (!is.list(votes[[i]]))
              votes[[i]] <- list()
            index <- as.character(predictions[i])
            if (is.null(votes[[i]][[index]]))
              votes[[i]][[index]] <- 0
            votes[[i]][[index]] <- votes[[i]][[index]] + weights[v]
          }
        }
      }
      if (private$task == "regression")
        return (rowSums(preds))
      res <- c()
      for (i in 1:n) {
        # TODO: if ex-aequos, random choice...
        ind_max <- which.max(unlist(votes[[i]]))
        pred_class <- names(votes[[i]])[ind_max]
        if (parse_numeric)
          pred_class <- as.numeric(pred_class)
        res <- c(res, pred_class)
      }
      res
    }
  ),
  private = list(
    data = NULL,
    target = NULL,
    task = NULL,
    gmodel = NULL,
    quality = NULL,
    run_res = NULL,
    get_testIndices = function(CV, v, n, shuffle_inds) {
      if (CV$type == "vfold") {
        first_index = round((v-1) * n / CV$V) + 1
        last_index = round(v * n / CV$V)
        test_indices = first_index:last_index
        if (CV$shuffle)
          test_indices <- shuffle_inds[test_indices]
      }
      else
        test_indices = sample(n, round(n * CV$test_size))
      test_indices
    },
    get_modelPerf = function(test_indices, p=0) {
      getOnePerf <- function(p) {
        model_pred <- private$gmodel$get(dataHO, targetHO, p)
        prediction <- model_pred(testX)
        perf <- private$quality(prediction, testY)
        list(model=model_pred, perf=perf)
      }
      dataHO <- private$data[-test_indices,]
      testX <- private$data[test_indices,]
      targetHO <- private$target[-test_indices]
      testY <- private$target[test_indices]
      # R will cast 1-dim matrices into vectors:
      if (!is.matrix(dataHO) && !is.data.frame(dataHO))
        dataHO <- as.matrix(dataHO)
      if (!is.matrix(testX) && !is.data.frame(testX))
        testX <- as.matrix(testX)
      if (p >= 1)
        # Standard CV: one model at a time
        return (getOnePerf(p)$perf)
      # Agghoo: loop on all models
      best_model = NULL
      best_perf <- -1
      for (p in 1:private$gmodel$nmodels) {
        model_perf <- getOnePerf(p)
        if (model_perf$perf > best_perf) {
          # TODO: if ex-aequos: models list + choose at random
          best_model <- model_perf$model
          best_perf <- model_perf$perf
        }
      }
      list(model=best_model, perf=best_perf)
    }
  )
)
